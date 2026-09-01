import math
import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

_EXT = None  # module-level cache; compiled once per Python process


def _dll_dirs():
    """On Windows, add CUDA and torch/lib to the DLL search path for the JIT ext."""
    handles = []
    if os.name != "nt":
        return handles
    candidates = []
    for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        p = os.environ.get(var)
        if p:
            candidates.append(os.path.join(p, "bin"))
    try:
        import torch as _torch

        candidates.append(os.path.join(os.path.dirname(_torch.__file__), "lib"))
    except ImportError:
        pass
    for d in candidates:
        if os.path.isdir(d):
            try:
                handles.append(os.add_dll_directory(d))
                logger.debug(f"Added DLL search dir: {d}")
            except OSError:
                pass
    return handles


def _load_ext():
    """JIT-compile the modular kernel files on first call; cache thereafter."""
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    cuda_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda"
    )
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    logger.info(
        "Compiling osd_0_cuda kernels — the first use in each Python environment "
        "takes a few minutes with no progress output (nvcc is running); later runs "
        "load instantly from the cache."
    )
    _load_ext._dll_handles = _dll_dirs()
    msvc_extra = ["-Xcompiler", "/Zc:preprocessor"] if os.name == "nt" else []
    _EXT = load(
        name="osd0_cuda_ext",
        sources=[
            os.path.join(cuda_dir, "osd0_kernel.cu"),
        ],
        extra_include_paths=[cuda_dir],  # so gf2.cuh / osd0.h resolve
        extra_cuda_cflags=["-O3", "-diag-suppress=221"] + msvc_extra,
        extra_cflags=["/Zc:preprocessor"] if os.name == "nt" else [],
        verbose=True,
    )
    logger.info("osd_0_cuda kernels compiled.")
    return _EXT


def _pack_H(H_np: np.ndarray, N: int, W: int) -> np.ndarray:
    """Bit-pack a dense {0,1} [M, N] matrix into uint64 [M, W] (column N = syndrome,
    left zero). Returns the int64 view so it can become a torch int64 tensor."""
    M = H_np.shape[0]
    packed = np.zeros((M, W), dtype=np.uint64)
    rows, cols = np.nonzero(H_np)
    word = (cols >> 6).astype(np.int64)
    bit = (cols & 63).astype(np.uint64)
    np.bitwise_or.at(packed, (rows, word), (np.uint64(1) << bit))
    return packed.view(np.int64)


def _gf2_rank(H_np: np.ndarray) -> int:
    """Rank over GF(2) via bit-packed elimination on the columns (host, once)."""
    M, N = H_np.shape
    W = (N + 63) >> 6
    rows = np.zeros((M, W), dtype=np.uint64)
    rr, cc = np.nonzero(H_np)
    np.bitwise_or.at(
        rows,
        (rr, (cc >> 6).astype(np.int64)),
        (np.uint64(1) << (cc & 63).astype(np.uint64)),
    )
    used = np.zeros(M, dtype=bool)
    rank = 0
    for c in range(N):
        wc, mc = c >> 6, np.uint64(1) << np.uint64(c & 63)
        piv = -1
        for r in range(M):
            if not used[r] and (rows[r, wc] & mc):
                piv = r
                break
        if piv < 0:
            continue
        used[piv] = True
        rank += 1
        col_bit = (rows[:, wc] & mc) != 0
        col_bit[piv] = False
        rows[col_bit] ^= rows[piv]
    return rank



class create(nn.Module):
    """
    OSD-0 post-decoder backed by bit-packed GF(2) CUDA kernels.

    Accepted YAML keys (under ``decoder:``):
        algorithm   : osd_0_cuda
        check_type  : hx | hz          (default: hx)
        dtype       : float32 | float64 | float16 | bfloat16  (default: float64)
        device:
          device_type : cuda           (only cuda is supported)
          device_idx  : int            (default: 0)
        force_per_step : bool          (optional; selects the modular debug path)
    """

    def __init__(self, decoder_cfg: dict, **kwargs) -> None:
        super().__init__()
        logger.info("Creating osd_0_cuda decoder.")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "osd_0_cuda requires a CUDA-capable GPU. Use osd_0 for CPU execution."
            )

        device_cfg = decoder_cfg.get("device", {})
        device_type = device_cfg.get("device_type", "cuda")
        if device_type != "cuda":
            logger.warning(
                f"osd_0_cuda only supports cuda; ignoring device_type='{device_type}'."
            )
        device_idx = device_cfg.get("device_idx", 0)
        if device_idx >= torch.cuda.device_count():
            logger.warning(f"device_idx={device_idx} exceeds available GPUs; using 0.")
            device_idx = 0
        self.device = torch.device(f"cuda:{device_idx}")

        dtype_str = decoder_cfg.get("dtype", "float64")
        if dtype_str not in {"float16", "bfloat16", "float32", "float64"}:
            logger.warning(f"Invalid dtype '{dtype_str}'; defaulting to float64.")
            dtype_str = "float64"
        self.dtype = torch.__dict__[dtype_str]

        self.check_type = decoder_cfg.get("check_type", "hx").lower()
        if self.check_type not in {"hx", "hz"}:
            logger.warning(f"Invalid check_type='{self.check_type}'; defaulting to hx.")
            self.check_type = "hx"

        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "osd_0_cuda requires a pre-loaded MatrixBundle via the `bundle` kwarg."
            )
        H_shape, _, V_c_col, H_matrix = bundle.select(self.check_type)
        self.H_shape = H_shape
        self.V_c_col = nn.Parameter(V_c_col.to(self.device), requires_grad=False)
        self.M, self.N = int(H_shape[0]), int(H_shape[1])
        self.W = ((self.N + 1) + 63) >> 6

        H_np = (H_matrix.detach().cpu().numpy() != 0).astype(np.uint8)
        self.A_rank = _gf2_rank(H_np)
        H_packed = torch.from_numpy(_pack_H(H_np, self.N, self.W)).contiguous()
        self.H_packed = nn.Parameter(H_packed.to(self.device), requires_grad=False)

        self.algo = "osd_0"
        self.num_max_iter = self.N  # matches osd_0.py:122
        self.batch_size = 1

        self._ext = _load_ext()
        smem_needed = self._ext.fused_smem_bytes(self.M, self.W)
        smem_limit = self._ext.fused_smem_limit()
        self._use_fused = smem_needed <= smem_limit and not decoder_cfg.get(
            "force_per_step", False
        )
        if not self._use_fused:
            logger.info(
                f"Using per-step kernel path "
                f"(fused needs {smem_needed} B shared memory, limit {smem_limit} B)."
            )

        # Block size for the fused kernel: cover M rows, round to a warp, cap 512.
        self._block_size = min(max(32 * math.ceil(self.M / 32), 64), 512)
        logger.info("osd_0_cuda decoder ready.")

    def forward(self, io_dict: dict) -> dict:
        """Re-solve BP's non-converged samples with OSD-0; return the full batch."""
        dev = self.device
        converge = io_dict["converge"]
        B = converge.shape[0]

        e_v = io_dict["e_v"].to(dtype=torch.uint8, device=dev).clone()
        iter_out = (
            io_dict["iter"].clone()
            if "iter" in io_dict
            else torch.zeros(B, dtype=torch.long, device=dev)
        )

        idx = (converge == 0).nonzero(as_tuple=True)[0].to(dev)
        if idx.numel() > 0:
            llr_sub = io_dict["llr"].to(dtype=self.dtype, device=dev)[idx]
            synd_sub = io_dict["synd"].to(device=dev)[idx]

            # Column order: most reliable handling matches osd_0.py:140
            # (stable ascending sort of the posterior LLR).
            _, order = torch.sort(llr_sub, dim=1, descending=False, stable=True)
            order = order.to(torch.int32).contiguous()
            synd_u8 = synd_sub.to(torch.uint8).contiguous()
            e_sub = torch.zeros(idx.numel(), self.N, dtype=torch.uint8, device=dev)

            if self._use_fused:
                self._ext.osd0_fused(
                    self.H_packed,
                    synd_u8,
                    order,
                    e_sub,
                    self.N,
                    self.A_rank,
                    self._block_size,
                )
            else:
                self._run_per_step(synd_u8, order, e_sub)

            e_v[idx] = e_sub
            iter_out = iter_out.to(dev)
            iter_out[idx] = self.num_max_iter

        io_dict.update(
            {
                "e_v": e_v,
                "iter": iter_out,
                "converge": torch.ones_like(converge),
            }
        )
        return io_dict

    def _run_per_step(self, synd_u8, order, e_sub):
        """Modular-kernel path: Python loop over column steps in global memory."""
        dev = self.device
        Bsub = synd_u8.shape[0]
        # Augmented [H | s] per sample: broadcast packed H, then set syndrome bit.
        aug = self.H_packed.unsqueeze(0).expand(Bsub, self.M, self.W).clone()
        wN, bN = self.N >> 6, self.N & 63
        aug[:, :, wN] |= synd_u8.to(torch.int64) << bN
        aug = aug.contiguous()

        row_pcol = torch.full((Bsub, self.M), -1, dtype=torch.int32, device=dev)
        pivot_row = torch.empty(Bsub, dtype=torch.int32, device=dev)
        for step in range(self.N):
            self._ext.osd_pivot(aug, order, row_pcol, pivot_row, step, self.N)
            self._ext.osd_eliminate(aug, order, pivot_row, step, self.N)
        self._ext.osd_solve(aug, row_pcol, e_sub, self.N)
