"""Independent, dense CPU message-passing oracle for the three CUDA decoders."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _graph(h):
    edges = h.nonzero().tolist()
    rows = [[e for e, (c, _) in enumerate(edges) if c == row] for row in range(len(h))]
    variables = [[e for e, (_, v) in enumerate(edges) if v == col] for col in range(h.shape[1])]
    row_ptr = torch.tensor([0] + list(torch.tensor([len(r) for r in rows]).cumsum(0).tolist()), dtype=torch.int32)
    col = torch.tensor([v for _, v in edges], dtype=torch.int32)
    vn = torch.full((h.shape[1] + 1, max(map(len, variables))), len(edges), dtype=torch.int32)
    for v, ids in enumerate(variables):
        vn[v, :len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return edges, rows, variables, row_ptr, col, vn


def _reference(h, base, syndrome, iterations, mode, alpha=1.0, interval=2, strength=4.0):
    edges, rows, variables, *_ = _graph(h)
    result = {k: [] for k in ("llr", "e_v", "iter", "converge")}
    events = {"erased": 0, "revived": 0, "biased": 0}
    for original, synd in zip(base, syndrome):
        prior = original.clone()
        outgoing = torch.stack([prior[v] for _, v in edges])
        incoming = torch.zeros_like(outgoing)
        previous_hard = prior <= 0
        flips = torch.zeros(len(prior), dtype=torch.int64)
        posterior = prior.clone()
        converged = False
        for iteration in range(1, iterations + 1):
            for c, ids in enumerate(rows):
                for e in ids:
                    others = outgoing[[other for other in ids if other != e]]
                    if mode == 2:
                        magnitude = others.abs().min()
                        sign = (-1) ** (int((others <= 0).sum()) + int(synd[c]))
                        incoming[e] = sign * magnitude
                    else:
                        product = torch.tanh(others / 2).prod() * (1 - 2 * int(synd[c]))
                        eps = torch.finfo(base.dtype).eps
                        incoming[e] = 2 * torch.atanh(product.clamp(-1 + eps, 1 - eps))
            for v, ids in enumerate(variables):
                total = incoming[ids].sum() if ids else base.new_tensor(0)
                posterior[v] = prior[v] + total / (alpha if mode == 0 else 1)
            hard = posterior <= 0
            flips += (hard != previous_hard).long()
            previous_hard = hard.clone()
            converged = torch.equal((h @ hard.long()) % 2, synd.long())
            if converged or iteration == iterations:
                break
            if mode == 1 and iteration % interval == 0:
                prior = original.clone()
                least = min(range(len(prior)), key=lambda v: (abs(float(posterior[v])), v))
                if int(flips.max()) > 0:
                    oscillating = min(range(len(prior)), key=lambda v: (-int(flips[v]), abs(float(posterior[v])), v))
                    prior[oscillating] = -strength
                    flips[oscillating] = 0
                prior[least] = -strength
                events["biased"] += 1
                for v, ids in enumerate(variables):
                    posterior[v] = prior[v] + (incoming[ids].sum() if ids else 0)
            for e, (_, v) in enumerate(edges):
                candidate = posterior[v] - incoming[e]
                if mode == 2:
                    candidate = prior[v].clone()
                    for other in variables[v]:
                        if other != e:
                            candidate += incoming[other]
                previous = outgoing[e].clone()
                if mode == 2 and candidate * previous < 0:
                    outgoing[e] = 0
                    events["erased"] += 1
                else:
                    outgoing[e] = candidate
                    if mode == 2 and previous == 0 and candidate != 0:
                        events["revived"] += 1
        result["llr"].append(posterior.clone())
        result["e_v"].append(hard.to(base.dtype))
        result["iter"].append(iteration)
        result["converge"].append(converged)
    return {k: torch.stack(v) if k in ("llr", "e_v") else torch.tensor(v) for k, v in result.items()}, events


def _inputs(dtype):
    # The first three checks form a cycle; odd parity around it is inconsistent,
    # forcing the stateful rules to run beyond their first intervention.
    h = torch.tensor([[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0]])
    generator = torch.Generator().manual_seed(73)
    base = torch.rand(9, 6, generator=generator, dtype=dtype) * 6 - 3
    base[0] = 2
    syndrome = torch.zeros(9, 4, dtype=torch.uint8)
    syndrome[1:, 0] = 1
    return h, base, syndrome


def _native(h, base, syndrome, iterations, mode, **kwargs):
    from syndrilla.decoder.bp_interventions import _load_ext

    *_, ptr, col, vn = _graph(h)
    values = _load_ext().decode(
        base.cuda(), syndrome.cuda(), ptr.cuda(), col.cuda(), vn.cuda(),
        iterations, mode, kwargs.get("alpha", 1.0),
        kwargs.get("interval", 2), kwargs.get("strength", 4.0),
    )
    return {k: v.cpu() for k, v in zip(("llr", "e_v", "iter", "converge"), values)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode,alpha", [(0, 1.0), (0, 1.7), (1, 1.0), (2, 1.0)])
@pytest.mark.parametrize("iterations", [1, 3, 12])
def test_native_matches_independent_oracle(dtype, mode, alpha, iterations):
    h, base, syndrome = _inputs(dtype)
    expected, events = _reference(h, base, syndrome, iterations, mode, alpha=alpha)
    actual = _native(h, base, syndrome, iterations, mode, alpha=alpha)
    torch.testing.assert_close(actual["llr"], expected["llr"], rtol=2e-5 if dtype == torch.float32 else 1e-9,
                               atol=2e-5 if dtype == torch.float32 else 1e-9)
    for key in ("e_v", "iter", "converge"):
        assert torch.equal(actual[key].reshape(expected[key].shape), expected[key]), key
    assert actual["converge"].flatten()[0] == 1
    assert not actual["converge"].flatten()[1:].any()
    if iterations == 12 and mode == 2:
        assert events["erased"] > 0 and events["revived"] > 0
    if iterations == 12 and mode == 1:
        assert events["biased"] > 0


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_batch_preserves_converged_samples_and_syndrome(mode):
    h, base, syndrome = _inputs(torch.float64)
    saved = syndrome.clone()
    batch = _native(h, base, syndrome, 12, mode, alpha=1.7)
    for shot in (0, 1, 8):
        single = _native(h, base[shot:shot + 1], syndrome[shot:shot + 1], 12, mode, alpha=1.7)
        for key in batch:
            torch.testing.assert_close(batch[key][shot:shot + 1], single[key])
    assert torch.equal(saved, syndrome)
    torch.testing.assert_close(batch["llr"][0], _native(h, base[:1], syndrome[:1], 1, mode, alpha=1.7)["llr"][0])


def test_memory_and_ots_have_separate_effects():
    h, base, syndrome = _inputs(torch.float64)
    bp = _native(h, base, syndrome, 12, 0)
    memory = _native(h, base, syndrome, 12, 0, alpha=1.7)
    ots = _native(h, base, syndrome, 12, 1, interval=2)
    ots_disabled = _native(h, base, syndrome, 12, 1, interval=20)
    torch.testing.assert_close(ots_disabled["llr"], bp["llr"])
    assert not torch.allclose(memory["llr"], bp["llr"])
    assert not torch.allclose(ots["llr"], bp["llr"])


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_degree_one_checks_enforce_supplied_syndrome(mode, dtype):
    h = torch.ones(1, 1, dtype=torch.int64)
    base = torch.tensor([[-2.0], [2.0]], dtype=dtype)
    syndrome = torch.tensor([[0], [1]], dtype=torch.uint8)
    result = _native(h, base, syndrome, 9, mode)
    assert torch.equal(result["e_v"], syndrome)
    assert result["converge"].all() and (result["iter"] == 1).all()
    assert not result["llr"].isnan().any()
    if mode != 2:
        assert result["llr"].isfinite().all()


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_empty_checks_and_isolated_variables(mode):
    h = torch.zeros(2, 3, dtype=torch.int64)
    base = torch.tensor([[1.0, -1.0, 0.0], [1.0, -1.0, 0.0]], dtype=torch.float64)
    syndrome = torch.tensor([[0, 0], [1, 0]], dtype=torch.uint8)
    result = _native(h, base, syndrome, 3, mode, interval=10)
    torch.testing.assert_close(result["llr"], base)
    assert torch.equal(result["e_v"], (base <= 0).to(torch.uint8))
    assert result["iter"].tolist() == [1, 3]
    assert result["converge"].tolist() == [1, 0]
