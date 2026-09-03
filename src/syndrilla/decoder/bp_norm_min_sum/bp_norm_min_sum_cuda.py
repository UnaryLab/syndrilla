import math
import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from syndrilla.decoder.decoder import RebatchSpeedup

_EXT = None  # module-level cache; compiled once per Python process


def _dll_dirs():
    """
    On Windows, add CUDA and torch/lib to the DLL search path so the compiled
    extension can find cudart and the ATen/c10 DLLs it was linked against.
    Returns a list of os.add_dll_directory context objects (kept alive by caller).
    """
    handles = []
    if os.name != "nt":
        return handles
    candidates = []
    # System CUDA toolkit bin
    for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        p = os.environ.get(var)
        if p:
            candidates.append(os.path.join(p, "bin"))
    # PyTorch lib (contains c10.dll, torch_cuda.dll, etc.)
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
    """JIT-compile bp_kernel.cu on first call; return the cached module thereafter."""
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    decoder_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_src = os.path.join(decoder_dir, "cuda", "bp_kernel.cu")
    # Compile only for the local GPU's architecture: faster first build and
    # silences torch's "TORCH_CUDA_ARCH_LIST is not set" warning.
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    logger.info(
        "Compiling bp_norm_min_sum_cuda kernels — the first use in each Python "
        "environment takes a few minutes with no progress output (nvcc is "
        "running); later runs load instantly from the cache."
    )
    # Keep DLL directory handles alive for the duration of this process.
    _load_ext._dll_handles = _dll_dirs()
    # /Zc:preprocessor — required by CCCL headers shipped with CUDA ≥ 12.x
    # when the host compiler is MSVC. Passed via -Xcompiler so nvcc forwards
    # it to cl.exe for CUDA compilation units.
    msvc_extra = ["-Xcompiler", "/Zc:preprocessor"] if os.name == "nt" else []
    _EXT = load(
        name="bp_nms_cuda_ext",
        sources=[kernel_src],
        extra_cuda_cflags=[
            "-O3",
            # --use_fast_math is intentionally omitted: it can affect double-
            # precision math (ldexp, etc.) in ways that corrupt the fused kernel
            # on some GPU/driver combinations.
            "-diag-suppress=221",  # float const assigned to double (HUGE_VAL on Windows)
        ]
        + msvc_extra,
        extra_cflags=["/Zc:preprocessor"] if os.name == "nt" else [],
        verbose=True,
    )
    logger.info("bp_norm_min_sum_cuda kernels compiled.")
    return _EXT



def _build_vn_adj(V_c_col_np: np.ndarray, N: int) -> tuple:
    """
    Build the variable→check adjacency tables needed by k_llr_update.

    V_c_col[c, k] gives the variable index for edge (c, k).  This function
    inverts that mapping: for each variable n, it returns the list of (c, k)
    pairs whose V_c_col entry equals n.

    Fully vectorised with numpy — no Python loops.

    Returns
    -------
    VN_adj_c : int64 ndarray [N_ext, VD]  — check index per var/slot (-1 = pad)
    VN_adj_k : int64 ndarray [N_ext, VD]  — degree slot k per var/slot (-1 = pad)
    VD       : int  — maximum variable-node degree
    """
    N_ext = N + 1

    # Locate all real (non-dummy) edges.
    c_idx, k_idx = np.where(V_c_col_np < N)  # shape [E]
    n_idx = V_c_col_np[c_idx, k_idx].astype(np.int64)

    if len(n_idx) == 0:
        return (
            np.full((N_ext, 1), -1, dtype=np.int64),
            np.full((N_ext, 1), -1, dtype=np.int64),
            1,
        )

    # Sort edges by (n, c, k) so each variable's group is contiguous.
    order = np.lexsort((k_idx, c_idx, n_idx))
    c_sorted = c_idx[order].astype(np.int64)
    k_sorted = k_idx[order].astype(np.int64)
    n_sorted = n_idx[order]

    # Maximum variable degree determines the second dimension.
    counts = np.bincount(n_sorted, minlength=N_ext)
    VD = int(counts.max())

    # CSR start pointer for each variable's edge group.
    ptr = np.zeros(N_ext + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts)

    slot = np.arange(len(n_sorted), dtype=np.int64) - ptr[n_sorted]

    VN_adj_c = np.full((N_ext, VD), -1, dtype=np.int64)
    VN_adj_k = np.full((N_ext, VD), -1, dtype=np.int64)
    VN_adj_c[n_sorted, slot] = c_sorted
    VN_adj_k[n_sorted, slot] = k_sorted

    return VN_adj_c, VN_adj_k, VD



class create(nn.Module):
    """
    BP Normalized Min-Sum decoder backed by modular CUDA kernels.

    Accepted YAML keys (under ``decoder:``):
        algorithm   : bp_norm_min_sum_cuda
        check_type  : hx | hz          (default: hx)
        max_iter    : int               (default: 50)
        dtype       : float32 | float64 | float16 | bfloat16  (default: float64)
        device:
          device_type : cuda            (only cuda is supported)
          device_idx  : int             (default: 0)
    """

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        nn.Module.__init__(self)
        logger.info("Creating bp_norm_min_sum_cuda decoder.")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "bp_norm_min_sum_cuda requires a CUDA-capable GPU. "
                "Use bp_norm_min_sum for CPU execution."
            )

        device_cfg = decoding_cfg.get("device", {})
        device_type = device_cfg.get("device_type", "cuda")
        if device_type != "cuda":
            logger.warning(
                f"bp_norm_min_sum_cuda only supports cuda; "
                f"ignoring device_type='{device_type}'."
            )
        device_idx = device_cfg.get("device_idx", 0)
        if device_idx >= torch.cuda.device_count():
            logger.warning(
                f"device_idx={device_idx} exceeds available GPUs; defaulting to 0."
            )
            device_idx = 0
        self.device = torch.device(f"cuda:{device_idx}")

        dtype_str = decoding_cfg.get("dtype", "float64")
        valid_dtypes = {"float16", "bfloat16", "float32", "float64"}
        if dtype_str not in valid_dtypes:
            logger.warning(f"Invalid dtype '{dtype_str}'; defaulting to float64.")
            dtype_str = "float64"
        self.dtype = torch.__dict__[dtype_str]

        self.max_iter = decoding_cfg.get("max_iter", 50)
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            logger.warning(f"Invalid max_iter={self.max_iter}; defaulting to 50.")
            self.max_iter = 50
        self.num_max_iter = self.max_iter

        self.check_type = decoding_cfg.get("check_type", "hx").lower()
        if self.check_type not in {"hx", "hz"}:
            logger.warning(f"Invalid check_type='{self.check_type}'; defaulting to hx.")
            self.check_type = "hx"

        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "bp_norm_min_sum_cuda requires a pre-loaded MatrixBundle "
                "passed as the 'bundle' keyword argument."
            )
        self.Hx_matrix = bundle.Hx_matrix
        self.Hz_matrix = bundle.Hz_matrix
        self.lx_matrix = bundle.lx_matrix
        self.lz_matrix = bundle.lz_matrix

        H_shape, V_c_row, V_c_col, H_matrix = bundle.select(self.check_type)
        self.H_shape = H_shape  # (M, N)
        self.N = H_shape[1]  # variable nodes (excludes dummy)
        self.N_ext = self.N + 1
        M_val, D = V_c_col.shape

        # Store V_c_col and V_c_row on the target device as non-trainable params.
        self.V_c_col = nn.Parameter(V_c_col.to(self.device), requires_grad=False)
        self.V_c_row = nn.Parameter(V_c_row.to(self.device), requires_grad=False)

        # Precomputed once at init; used every iteration by k_llr_update.
        V_c_col_np = V_c_col.cpu().numpy()
        adj_c, adj_k, self.VD = _build_vn_adj(V_c_col_np, self.N)

        self.VN_adj_c = nn.Parameter(
            torch.from_numpy(adj_c).to(self.device), requires_grad=False
        )
        self.VN_adj_k = nn.Parameter(
            torch.from_numpy(adj_k).to(self.device), requires_grad=False
        )

        # Must cover M*D edges (stride-loop handles the rest).
        # Round up to the next multiple of 32 (warp), cap at 512.
        min_threads = M_val * D
        self._block_size = min(max(32 * math.ceil(min_threads / 32), 64), 512)

        self.algo = "bp_norm_min_sum"
        self.batch_size = 1  # updated in forward()

        self.cap = RebatchSpeedup.from_cfg(decoding_cfg.get("rebatch_speedup"))
        self.cap_bypass = False  # set by main: True -> decode this batch uncapped
        self.cap_active_last = False  # set per forward: True if the cap was applied

        self._ext = _load_ext()

        smem_needed = self._ext.fused_smem_bytes(
            M_val, D, self.N_ext, self.VD, self.dtype.itemsize
        )
        smem_limit = self._ext.fused_smem_limit()
        self._use_fused = smem_needed <= smem_limit and not decoding_cfg.get(
            "force_per_step", False
        )
        if not self._use_fused:
            logger.info(
                f"Using per-step kernel path "
                f"(fused needs {smem_needed} B shared memory, limit {smem_limit} B)."
            )
        logger.info("bp_norm_min_sum_cuda decoder ready.")

    def forward(self, io_dict: dict) -> dict:
        """
        Run BP Normalized Min-Sum decoding via the fused on-device kernel.

        The entire iteration loop (VN update → CN update → LLR update →
        hard decision → syndrome check → convergence test) runs inside a
        single CUDA kernel launch — no Python loop, no GPU→CPU syncs.

        Parameters (from io_dict)
        -------------------------
        synd  : [B, M]  binary syndrome tensor
        llr0  : [B, N]  initial channel LLR (log P(0)/P(1) per qubit)

        Updates io_dict with
        --------------------
        e_v      : [B, N]   hard-decision error estimate
        llr      : [B, N]   final per-variable LLR
        iter     : [B]      iteration at which each sample converged (max_iter if not)
        converge : [B]      1 if converged, 0 otherwise
        """
        dev = self.device
        # .contiguous() is load-bearing: every kernel indexes raw data pointers
        # assuming row-major layout. A transposed-stride syndrome (e.g. built
        # via (H @ e.T).T) would otherwise be read as other samples' bits.
        syndrome = io_dict["synd"].to(dtype=self.dtype, device=dev).contiguous()
        B, M = syndrome.shape
        self.batch_size = B

        # Append dummy column (∞) so variable index N always gives ∞ LLR.
        llr0 = io_dict["llr0"].to(dtype=self.dtype, device=dev).contiguous()
        dummy_col = torch.full((B, 1), float("inf"), dtype=self.dtype, device=dev)
        u_init = torch.cat([llr0, dummy_col], dim=1)  # [B, N_ext]

        # syndrome_neg_bc[b, c] = +1 if syndrome[b,c]==0 else −1
        syndrome_neg_bc = torch.where(
            syndrome == 0.0,
            torch.ones_like(syndrome),
            -torch.ones_like(syndrome),
        )

        # Output buffers.
        e_out = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        l_out = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        num_iters = torch.zeros(B, dtype=torch.int64, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        self.cap_active_last = bool(
            self.cap is not None and self.cap.done and not self.cap_bypass
        )
        use_fused = self._use_fused

        if use_fused:
            conv_count, stop_count = None, 0
            if self.cap_active_last:
                conv_count = torch.zeros(1, dtype=torch.int32, device=dev)
                stop_count = int(math.ceil(self.cap.frac * B))
            self._ext.bp_nms_fused(
                u_init,
                syndrome_neg_bc,
                self.V_c_col,
                self.VN_adj_c,
                self.VN_adj_k,
                e_out,
                l_out,
                num_iters,
                converges,
                self.N,
                self.VD,
                self.max_iter,
                self._block_size,
                conv_count,
                stop_count,
            )
        else:
            HOST_CHECK_EVERY = 8
            cap_frac = self.cap.frac if self.cap_active_last else None
            D = int(self.V_c_col.shape[1])
            a_v2c = torch.zeros(B, M, D, dtype=self.dtype, device=dev)
            b_c2v = torch.zeros(B, M, D, dtype=self.dtype, device=dev)
            l_v = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
            e_v = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
            s_est = torch.zeros(B, M, dtype=self.dtype, device=dev)
            num_iters.fill_(-1)

            for i in range(1, self.max_iter + 1):
                beta = 1.0 - 2.0 ** (-i)
                if i == 1:
                    self._ext.init_messages(u_init, self.V_c_col, a_v2c)
                else:
                    self._ext.vn_update(l_v, b_c2v, self.V_c_col, a_v2c, self.N)
                self._ext.cn_update(
                    a_v2c, syndrome_neg_bc, self.V_c_col, b_c2v, beta, self.N
                )
                self._ext.llr_update(
                    u_init, b_c2v, self.VN_adj_c, self.VN_adj_k, l_v, self.VD
                )
                l_v[:, -1] = float("inf")
                self._ext.hard_decision(l_v, e_v)
                self._ext.syndrome_est(e_v, self.V_c_col, s_est, self.N)
                self._ext.convergence_update(
                    s_est,
                    syndrome,
                    e_v,
                    l_v,
                    e_out,
                    l_out,
                    num_iters,
                    converges,
                    i,
                )
                if cap_frac is not None:
                    # cap active: count converged each iteration (host sync) and stop
                    # once the learned fraction is reached or every sample converges.
                    n_conv = int((num_iters != -1).sum())
                    if n_conv >= cap_frac * B or n_conv == B:
                        break
                elif i % HOST_CHECK_EVERY == 0 and not (num_iters == -1).any().item():
                    break

            not_conv = num_iters == -1
            if not_conv.any().item():
                e_out[not_conv] = e_v[not_conv]
                l_out[not_conv] = l_v[not_conv]
                num_iters[not_conv] = i

        # warm-up: observe this batch's stop-iteration distribution (decides k + the
        # cap percentile). Skipped once warm-up is done or when main bypassed the cap.
        if self.cap is not None and not self.cap.done and not self.cap_bypass:
            self.cap.observe(num_iters, self.max_iter, B)

        # Strip dummy column and return.
        io_dict.update(
            {
                "e_v": e_out[:, :-1],
                "iter": num_iters,
                "llr": l_out[:, :-1],
                "converge": converges,
            }
        )
        return io_dict
