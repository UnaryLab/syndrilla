"""Save frozen reference outputs for tests/test_bp_variants_pytorch_equiv.py.

usage: python tests/freeze_bp_variant_pytorch_refs.py <outdir> [<variant> ...]
Saves <outdir>/<variant>/<case>_<dtype>/{inputs.pt,outputs.pt,config.json} for
every variant (default: all), case and dtype in the test module, decoding on CPU.
The stim_d5 and surface10 inputs are those of tests/freeze_bp_variant_refs.py,
whose stim_d5 shots are drawn on CUDA, so this script needs a GPU; the
surface10_depol inputs (bp4) are drawn on CPU."""
import json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import test_bp_variants_cuda_equiv as C
import test_bp_variants_pytorch_equiv as T
from syndrilla.utils import read_yaml

YAML = {**C.VARIANTS, "bp_sum_prod": "bp_sum_prod_hx", "bp4": "bp4"}


def decoder_cfg(variant, dtype):
    path = os.path.join(T.ROOT, "examples", "alist", f"{YAML[variant]}.decoding.yaml")
    return {
        **read_yaml(path)["decoding"],
        "device": T.CPU,
        "dtype": dtype,
        "check_type": "hx",
        "force_pytorch": True,
    }


def depol_inputs(dtype):
    """Depolarizing p=0.03 on surface10, perfect syndrome measured through bp4."""
    from syndrilla.decoder import create_decoder
    from syndrilla.error_model import create_error_model
    from syndrilla.syndrome import create_syndrome

    bundle = T.bundle("surface10", dtype)
    bp4 = create_decoder(cfg=decoder_cfg("bp4", dtype), bundle=bundle)[0]
    torch.manual_seed(T.SEED)
    np.random.seed(T.SEED)
    random.seed(T.SEED)
    em = create_error_model(
        cfg={"model": "depol", "rate": C.P_ALIST, "device": {"device_type": "cpu"}}
    )
    z = torch.zeros(C.B, bundle.select("hx")[0][1], dtype=getattr(torch, dtype))
    _, dl = em.inject_error(z, C.B)
    e, llr0, _ = next(iter(dl))
    return (
        create_syndrome(cfg={"measure": "perfect"}).measure_syndrome(
            e, getattr(bp4, "decoder", bp4)
        ),
        llr0,
    )


out = sys.argv[1]
variants = sys.argv[2:] or list(T.VARIANTS) + ["bp4"]
for variant, case in T.CASES:
    if variant not in variants:
        continue
    for dtype in C.DTYPES:
        d = os.path.join(out, variant, f"{case}_{dtype}")
        os.makedirs(d, exist_ok=True)
        if case == "surface10_depol":
            synd, llr0 = depol_inputs(dtype)
        else:
            _, synd, llr0 = C.bundle_and_inputs(case, dtype)
        synd, llr0 = synd.cpu(), llr0.cpu()
        cfg = decoder_cfg(variant, dtype)
        t = time.time()
        res = T.decode(cfg, case, dtype, synd, llr0)
        torch.save({"synd": synd, "llr0": llr0}, f"{d}/inputs.pt")
        torch.save(res, f"{d}/outputs.pt")
        json.dump(
            {
                "variant": variant,
                "case": case,
                "dtype": dtype,
                "B": C.B,
                "yaml": f"examples/alist/{YAML[variant]}.decoding.yaml",
                "decoder_cfg": cfg,
                "seeds": {
                    "torch": T.SEED,
                    "numpy": T.SEED,
                    "python_random": T.SEED,
                    "sobol": "SobolEngine(dimension=1, scramble=False), fresh per forward",
                },
                "torch": torch.__version__,
                "device": "cpu",
            },
            open(f"{d}/config.json", "w"),
            indent=1,
        )
        print(
            f"{variant} {case} {dtype}: {time.time()-t:.1f}s"
            f" conv {int(res['converge'].sum())}/{C.B}",
            flush=True,
        )
