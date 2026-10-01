"""Save frozen reference outputs for tests/test_bp_variants_cuda_equiv.py.

usage: python tests/freeze_bp_variant_refs.py <variant> <outdir>
Saves <outdir>/<variant>/<case>_<dtype>/{inputs.pt,outputs.pt,config.json} for
every case and dtype in the test module."""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import test_bp_variants_cuda_equiv as T

variant, out = sys.argv[1], sys.argv[2]
for case in T.CASES:
    for dtype in T.DTYPES:
        d = os.path.join(out, variant, f"{case}_{dtype}")
        os.makedirs(d, exist_ok=True)
        bundle, synd, llr0 = T.bundle_and_inputs(case, dtype)
        cfg = T.decoder_cfg(variant, dtype)
        t = time.time()
        res = T.decode(variant, cfg, bundle, synd, llr0)
        torch.save({"synd": synd.cpu(), "llr0": llr0.cpu()}, f"{d}/inputs.pt")
        torch.save(res, f"{d}/outputs.pt")
        src = (
            {
                "kind": "stim",
                "code": "surface_code:rotated_memory_x",
                "distance": 5,
                "rounds": 5,
                "noise": {k: T.P_STIM for k in T.NOISE},
                "shot_seed": T.SEED,
            }
            if case == "stim_d5"
            else {
                "kind": "alist",
                "matrix_yaml": "examples/alist/surface_10.matrix.yaml",
                "check": "hx",
                "bsc_p": T.P_ALIST,
                "shot_seed": T.SEED,
            }
        )
        json.dump(
            {
                "variant": variant,
                "case": case,
                "dtype": dtype,
                "B": T.B,
                "seeds": {
                    "torch": T.SEED,
                    "torch_cuda": T.SEED,
                    "numpy": T.SEED,
                    "python_random": T.SEED,
                    "sobol": "SobolEngine(dimension=1, scramble=False), fresh per forward, no seed",
                },
                "bundle": src,
                "yaml": f"examples/alist/{T.VARIANTS[variant]}.decoding.yaml",
                "decoder_cfg": cfg,
                "deterministic": True,
                "note": "",
            },
            open(f"{d}/config.json", "w"),
            indent=1,
        )
        print(
            f"{variant} {case} {dtype}: {time.time()-t:.1f}s conv {int(res['converge'].sum())}/{T.B} "
            f"iter max {int(res['iter'].max())} shapes {[(k, tuple(v.shape), str(v.dtype)) for k, v in res.items()]}",
            flush=True,
        )
