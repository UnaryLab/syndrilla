"""BP with guided decimation, Yao et al., arXiv:2312.10950v2.

Hanwen Yao, Waleed Abu Laban, Christian Hager, Alexandre Graell i Amat,
Henry D. Pfister, "Belief Propagation Decoding of Quantum LDPC Codes with
Guided Decimation", Algorithm 1, Sec. 3.1 and 4.3-4.4.
https://arxiv.org/abs/2312.10950

Messages initialize once and persist across BP rounds. Reliability is the
absolute posterior (eq. 26); eq. 36 fixes the chosen channel prior to +llr_max
only for a strictly positive posterior, otherwise to -llr_max. This resolves
Algorithm 1 line 11's conflicting >= sign at zero using eq. 27 and eq. 36.
Finite priors stay fixed in later rounds; they are not infinite hard pins.
Eq. 23 clips each variable-to-check message to [-saturation, saturation],
with saturation default 25. Priors, posteriors and check messages are not clipped.

Choices where the paper is silent: reliability ties use the lowest node index;
the total finite budget runs complete BP rounds only, capped by max_rounds
and the number of nodes as in Algorithm 1. The default max_iter 50 gives five
rounds at round_iters 10; the paper runs n rounds.
No unused decimation follows the final BP update. Output rounds counts actual
prior edits, not completed BP blocks. early_termination false checks matches
only at round ends, and still stops a shot when it matches there. The base
sum-product implementation clamps probabilities by the dtype epsilon.
Random order (Sec. 4.4) draws uniformly from undecimated nodes. As in lottery,
Sobol is unscrambled and decimates one fixed node sequence shared by the batch;
system draws per shot with the device RNG. CPU and CUDA system RNG streams need
not match.
Float32 backends can differ on unconverged shots: argmax |posterior| and the
eq. 36 sign near zero can turn base sum-product drift into a different pick
or prior sign. With matched random draws, float64 is bit-exact across backends.
"""
import math
import types

import torch

from syndrilla.decoder.bp_sum_prod.bp_sum_prod import _cn_q, create as SumProduct
from syndrilla.decoder.knobs import reject_unknown_keys


def _validate(cfg):
    for key, default in (("round_iters", 10), ("max_iter", 50)):
        value = cfg.get(key, default)
        if type(value) is not int or not 0 < value < 2**31:
            raise ValueError(f"{key} must be a positive int below 2^31")
    if cfg.get("max_iter", 50) < cfg.get("round_iters", 10):
        raise ValueError("max_iter must be at least round_iters")
    if "max_rounds" in cfg and (type(cfg["max_rounds"]) is not int or not 0 < cfg["max_rounds"] < 2**31):
        raise ValueError("max_rounds must be a positive int below 2^31")
    for key in ("llr_max", "saturation"):
        value = cfg.get(key, 25.)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if cfg.get("order", "reliability") not in ("reliability", "random"):
        raise ValueError("order must be reliability or random")
    if cfg.get("random_machine", "sobol") not in ("sobol", "system"):
        raise ValueError("random_machine must be sobol or system")
    if cfg.get("dtype", "float64") not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if cfg.get("check_type", "hx") not in ("hx", "hz"):
        raise ValueError("check_type must be hx or hz")
    if type(cfg.get("early_termination", True)) is not bool:
        raise ValueError("early_termination must be a bool")


def _step(posterior, messages, prior, syndrome_odd, col, vn_adj, mask_dummy, saturation):
    batch, width = posterior.shape
    checks, degree = mask_dummy.shape
    outgoing = posterior[:, col].view(batch, checks, degree) - messages
    outgoing.clamp_(-saturation, saturation).masked_fill_(mask_dummy, float("inf"))
    q = _cn_q(outgoing, syndrome_odd)
    incoming = torch.log((1.0 - q) / q).masked_fill(mask_dummy, 0.0)
    flat = torch.cat([incoming.view(batch, -1), incoming.new_zeros(batch, 1)], 1)
    gathered = flat[:, vn_adj].view(batch, -1, width)
    out = prior
    for slot in gathered.unbind(1):
        out = out + slot
    hard = out <= 0
    estimate = hard.to(torch.uint8)[:, col].view(batch, checks, degree).sum(2, dtype=torch.uint8) & 1
    return incoming, out, hard.to(posterior.dtype), estimate.to(posterior.dtype)


class create(SumProduct):
    _config_keys = ("round_iters", "llr_max", "max_rounds", "order", "random_machine", "saturation")
    _config_algorithm = "bp_gd"

    def __init__(self, cfg, **kwargs):
        reject_unknown_keys(cfg, self._base_config_keys.union(self._config_keys),
                            cfg.get("algorithm", self._config_algorithm))
        _validate(cfg)
        super().__init__(cfg, **kwargs)
        self._init_gd(cfg)
        if self.compile:
            step = types.FunctionType(_step.__code__.replace(), _step.__globals__, _step.__name__)
            self._step = torch.compile(step)

    def _init_gd(self, cfg):
        self.round_iters = cfg.get("round_iters", 10)
        self.llr_max = cfg.get("llr_max", 25.)
        self.saturation = cfg.get("saturation", 25.)
        if self.llr_max > torch.finfo(self.dtype).max:
            raise ValueError("llr_max must fit dtype")
        self.max_rounds = cfg.get("max_rounds", self.H_shape[1])
        self.order = cfg.get("order", "reliability")
        self.random_machine = cfg.get("random_machine", "sobol")
        self._gd_limit = min(self.max_rounds, self.H_shape[1], self.max_iter // self.round_iters) * self.round_iters
        self.algo = "bp_gd"

    def _new_messages(self, batch):
        return torch.zeros((batch, *self.V_c_col.shape), device=self.device, dtype=self.dtype)

    def _bp_step(self, messages, posterior, prior, syndrome_odd):
        if self.compile:
            return self._step(posterior, messages, prior, syndrome_odd,
                              self.V_c_col.flatten(), self.vn_adj, self.mask_dummy, self.saturation)
        if self._gd_buffers is None or not self.reuse_buffers:
            self._gd_buffers = self._work_buffers(posterior)
        flat, incoming, out, _, outgoing, gathered = self._gd_buffers
        outgoing = self.v2c(posterior, outgoing).sub_(messages)
        outgoing.clamp_(-self.saturation, self.saturation).masked_fill_(self.mask_dummy, float("inf"))
        self.cn_update(outgoing, syndrome_odd, incoming)
        self.llr_update(prior, self.c2v(flat, out, gathered), out)
        hard = (out <= 0).to(self.dtype)
        return incoming, out, hard, self.syndrome_estimation(hard)

    def _posterior_from_messages(self, prior, messages, out):
        flat = torch.cat((messages.flatten(1), messages.new_zeros((len(prior), 1))), 1)
        incoming = flat[:, self.vn_adj].view(len(prior), -1, prior.shape[1])
        return SumProduct.llr_update(self, prior, incoming, out)

    def _random_values(self, rows):
        if self.random_machine == "sobol":
            return self._gd_draws[self.i // self.round_iters - 1].expand(len(rows))
        return torch.rand(len(rows), device=self.device, dtype=self.dtype)

    def _decimate(self, rows, posterior, prior, messages, active):
        pool = self._gd_undecimated[rows]
        valid = active & pool.any(1)
        if self.order == "reliability":
            nodes = posterior[:, :-1].abs().masked_fill(~pool, -float("inf")).argmax(1)
        else:
            ranks = (self._random_values(rows) * pool.sum(1)).long()
            nodes = (pool.long().cumsum(1) > ranks[:, None]).to(torch.int8).argmax(1)
        selected = valid.nonzero().flatten()
        nodes = nodes[selected]
        values = posterior.new_full((len(selected),), self.llr_max)
        values.masked_fill_(posterior[selected, nodes] <= 0, -self.llr_max)
        prior[selected, nodes] = values
        self._gd_current[rows[selected], nodes] = values
        self._gd_undecimated[rows[selected], nodes] = False
        self._gd_rounds[rows[selected]] += 1
        self.last_gd_nodes = torch.full((len(rows),), -1, device=self.device, dtype=torch.long)
        self.last_gd_nodes[selected] = nodes
        self._posterior_from_messages(prior, messages, posterior)

    @torch.no_grad()
    def forward(self, io_dict):
        channel = io_dict["llr0"].to(self.device, self.dtype)
        syndrome = io_dict["synd"].to(self.device, self.dtype)
        if channel.ndim != 2 or channel.shape[1] != self.H_shape[1]:
            raise ValueError("llr0 must have shape [batch, variables]")
        b, n = channel.shape
        if syndrome.shape != (b, self.H_shape[0]):
            raise ValueError("synd must match the batch and check dimensions")
        self._gd_current = channel.clone()
        self._gd_undecimated = torch.ones((b, n), dtype=torch.bool, device=self.device)
        self._gd_rounds = torch.zeros(b, dtype=torch.long, device=self.device)
        self.last_gd_nodes = None
        self._gd_draws = torch.quasirandom.SobolEngine(1, scramble=False).draw(
            self._gd_limit // self.round_iters, dtype=torch.float32).flatten().to(self.device, self.dtype)
        prior = torch.cat((channel, channel.new_full((b, 1), float("inf"))), 1)
        posterior, messages = prior.clone(), self._new_messages(b)
        rows = torch.arange(b, device=self.device)
        active = torch.ones(b, device=self.device, dtype=torch.bool)
        result = channel.clone()
        iterations = torch.zeros(b, dtype=torch.long, device=self.device)
        converged = torch.zeros_like(iterations)
        cap_frac = (self.cap.frac if self.cap is not None and self.cap.done and not self.cap_bypass else None)
        self.cap_active_last = False
        self._gd_buffers = None
        self.i = 0
        for i in range(1, self._gd_limit + 1):
            if not bool(active.any()):
                break
            if 0 < int(active.sum()) < self.compact_frac * len(rows):
                keep = active.nonzero().flatten()
                rows, prior, posterior, messages = [value[keep] for value in (rows, prior, posterior, messages)]
                syndrome, active = syndrome[keep], active[keep]
                self._gd_buffers = None
            self.i, self.batch_size = i, len(rows)
            messages, posterior, hard, estimate = self._bp_step(messages, posterior, prior, (syndrome != 0).unsqueeze(2))
            result[rows[active]] = posterior[active, :-1]
            iterations[rows[active]] = i
            boundary = i % self.round_iters == 0
            if self.early_termination or boundary:
                done = active & (estimate == syndrome).all(1)
                converged[rows[done]] = 1
                active[done] = False
            if cap_frac is not None and int(converged.sum()) >= cap_frac * b:
                self.cap_active_last = bool(active.any() and i < self._gd_limit)
                break
            if boundary and i < self._gd_limit:
                self._decimate(rows, posterior, prior, messages, active)
        self.batch_size = b
        if self.cap is not None and not self.cap.done and not self.cap_bypass and b:
            self.cap.observe(iterations, self._gd_limit, b)
        io_dict.update(llr=result, e_v=(result <= 0).to(self.dtype), converge=converged,
                       iter=iterations, rounds=self._gd_rounds.clone(),
                       defer=(converged == 0) if self.cap_active_last else torch.zeros(b, device=self.device, dtype=torch.bool))
        return io_dict
