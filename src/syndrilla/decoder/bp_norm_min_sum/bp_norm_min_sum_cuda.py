import math
import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from syndrilla.decoder.decoder import RebatchSpeedup
from syndrilla.decoder.knobs import knob

_EXT = None  # module-level cache; compiled once per Python process
PERSISTENT_THREADS = (
    1024  # threads per block (one block per sample) on the persistent path
)
# Largest check count M for which large batches (B >= 4 x SM count) take the
# persistent path. Stim rotated_memory_x, rounds = d, B = 1000, RTX 4090, f64 and
# f32, p = 3e-3 and 1e-3: persistent is faster at M = 720 (d=9) and 1320 (d=11)
# and slower at M = 2184 (d=13), 3360 (d=15) and 19656 (d=27, B = 512).
# Tuned on one RTX 4090 at B = 1000; the d=11 f64 margin is 1.02x, so other GPUs may differ.
PERSISTENT_MAX_M = 1320


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
        "Compiling bp_norm_min_sum_cuda kernels: the first use in each Python "
        "environment takes a few minutes with no progress output (nvcc is "
        "running); later runs load instantly from the cache."
    )
    # Keep DLL directory handles alive for the duration of this process.
    _load_ext._dll_handles = _dll_dirs()
    # /Zc:preprocessor: required by CCCL headers shipped with CUDA ≥ 12.x
    # when the host compiler is MSVC. Passed via -Xcompiler so nvcc forwards
    # it to cl.exe for CUDA compilation units.
    msvc_extra = ["-Xcompiler", "/Zc:preprocessor"] if os.name == "nt" else []
    _EXT = load(
        name="bp_nms_cuda_ext",
        sources=[kernel_src],
        extra_cuda_cflags=[
            "-O3",
            # --use_fast_math is intentionally omitted: it can affect double-
            # precision math (ldexp, etc.) in ways that corrupt the kernels
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
    Build the variable→check adjacency tables. The CUDA decoders
    (bp_norm_min_sum_cuda, relay_bp_cuda, bp_branch_assisted_cuda) pass them to
    _build_csr, which turns them into VN_eid; the PyTorch relay_bp and
    bp_branch_assisted decoders flatten them into their vn_adj gather index.

    V_c_col[c, k] gives the variable index for edge (c, k).  This function
    inverts that mapping: for each variable n, it returns the list of (c, k)
    pairs whose V_c_col entry equals n.

    Fully vectorised with numpy: no Python loops.

    Returns
    -------
    VN_adj_c : int64 ndarray [N_ext, VD]  check index per var/slot (-1 = pad)
    VN_adj_k : int64 ndarray [N_ext, VD]  degree slot k per var/slot (-1 = pad)
    VD       : int  maximum variable-node degree
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


def _build_csr(
    V_c_col_np: np.ndarray, N: int, adj_c: np.ndarray, adj_k: np.ndarray
) -> tuple:
    """
    CSR edge list of the real edges of V_c_col (dummy edges n == N dropped) in
    (c, k) order, for the per-step kernels.

    Returns
    -------
    row_ptr : int32 ndarray [M + 1]: edges of check c are row_ptr[c] .. row_ptr[c+1]-1
    col     : int32 ndarray [nnz]: variable index per edge
    VN_eid  : int32 ndarray [N_ext, VD]: flat edge id per var/slot, in the same
              (c, k) order as adj_c/adj_k; padding points at nnz, the always-zero
              last slot of the [B, nnz + 1] edge buffer
    """
    real = V_c_col_np < N
    row_ptr = np.zeros(V_c_col_np.shape[0] + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum(real.sum(axis=1))
    nnz = int(row_ptr[-1])
    if nnz >= 2**31:
        raise ValueError(
            f"CSR edge layout needs nnz < 2^31 for int32 indices, got nnz = {nnz}"
        )
    col = V_c_col_np[real]  # row-major boolean indexing keeps (c, k) order
    eid = np.full(V_c_col_np.shape, nnz, dtype=np.int64)
    eid[real] = np.arange(nnz)
    VN_eid = np.where(adj_c >= 0, eid[adj_c, adj_k], nnz)
    return row_ptr.astype(np.int32), col.astype(np.int32), VN_eid.astype(np.int32)


class create(nn.Module):
    """
    BP Normalized Min-Sum decoder backed by modular CUDA kernels.

    Accepted YAML keys (under ``decoder:``):
        algorithm   : bp_norm_min_sum_cuda
        check_type  : hx | hz          (default: hx)
        max_iter    : int               (default: 50)
        dtype       : float32 | float64 | float16 | bfloat16  (default: float64)
        force_per_step : bool           (default: false; alias of persistent_kernel:
                      false, used when persistent_kernel is absent)
        persistent_kernel : auto | true | false  (default: auto; true takes the
                      persistent kernel whenever it can run, false never)
        host_check_every : int          (default: 8; per-step host early-exit
                      check every this many iterations, 0 never)
        skip_converged : bool           (default: true; false runs every sample
                      to the batch end, returns each converged sample's iterate at
                      its convergence iteration, and takes the per-step path)
        warp_per_check : bool           (default: true; false runs each check row
                      on one thread, a serial scan in k order; per-step path)
        fuse_vn     : bool              (default: true; false writes the VN->CN
                      messages to an a_v2c edge buffer in a separate kernel before
                      the check update; per-step path)
        f64_int_compare : bool          (default: true; false uses the
                      floating-point compares for double in the check update)
        edge_layout : csr | padded      (default: csr; padded runs the kernels on
                      the [M, D] check rows with dummy edges)
        vn_gather   : bool              (default: true; false sums each variable's
                      check messages with one atomicAdd per edge, in no fixed
                      order, an ablation baseline; per-step path)
        The quantized subclasses run the default kernels for these five knobs
        (warp_per_check, fuse_vn and vn_gather still force the per-step path).
        pruning_opt, fusion_opt, mapping_opt, gather_opt, memory_opt, rebatch_opt :
                      bool (default: true; false sets the group's member knobs to
                      their GROUP_OFF values in syndrilla.decoder.knobs, an
                      explicit member key wins)
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

        H_shape, V_c_row, V_c_col, H_matrix = bundle.select(
            self.check_type
        )
        self.H_shape = H_shape  # (M, N)
        self.N = H_shape[1]  # variable nodes (excludes dummy)
        self.N_ext = self.N + 1

        # Store V_c_col and V_c_row on the target device as non-trainable params.
        self.V_c_col = nn.Parameter(V_c_col.to(self.device), requires_grad=False)
        self.V_c_row = nn.Parameter(V_c_row.to(self.device), requires_grad=False)

        # Variable→check adjacency: _build_csr builds VN_eid from it.
        V_c_col_np = V_c_col.cpu().numpy()
        adj_c, adj_k, self.VD = _build_vn_adj(V_c_col_np, self.N)

        self.VN_adj_c = nn.Parameter(
            torch.from_numpy(adj_c).to(self.device), requires_grad=False
        )
        self.VN_adj_k = nn.Parameter(
            torch.from_numpy(adj_k).to(self.device), requires_grad=False
        )

        # CSR edge layout used by the per-step and persistent paths.
        row_ptr, col, vn_eid = _build_csr(V_c_col_np, self.N, adj_c, adj_k)
        self.nnz = len(col)
        self.row_ptr = nn.Parameter(
            torch.from_numpy(row_ptr).to(self.device), requires_grad=False
        )
        self.col = nn.Parameter(
            torch.from_numpy(col).to(self.device), requires_grad=False
        )
        self.VN_eid = nn.Parameter(
            torch.from_numpy(vn_eid).to(self.device), requires_grad=False
        )

        edge_layout = str(knob(decoding_cfg, "edge_layout", "csr")).lower()
        if edge_layout not in {"csr", "padded"}:
            raise ValueError(f"edge_layout must be csr or padded, got {edge_layout!r}")
        self._padded = edge_layout == "padded"
        if self._padded:
            # Padded edge layout: col_pad [M, D] with the dummy index N, and
            # VN_eid_pad with edge ids c * D + k, padded with M * D.
            M, D = V_c_col_np.shape
            if M * D >= 2**31:
                raise ValueError(f"padded edge layout needs M * D < 2^31, got {M * D}")
            self.col_pad = nn.Parameter(
                torch.from_numpy(V_c_col_np.astype(np.int32)).to(self.device),
                requires_grad=False,
            )
            eid_pad = np.where(adj_c >= 0, adj_c * D + adj_k, M * D).astype(np.int32)
            self.VN_eid_pad = nn.Parameter(
                torch.from_numpy(eid_pad).to(self.device), requires_grad=False
            )

        self.algo = "bp_norm_min_sum"
        self.batch_size = 1  # updated in forward()

        self.cap = RebatchSpeedup.from_cfg(decoding_cfg)
        self.cap_bypass = False  # set by main: True -> decode this batch uncapped
        self.cap_active_last = False  # set per forward: True if the cap stopped early

        self._ext = _load_ext()

        # Persistent-path limits, fixed per device and dtype: the SM count and how
        # many persistent blocks the device holds at once (the capped launch needs
        # every sample's block co-resident).
        persistent = knob(decoding_cfg, "persistent_kernel", "auto")
        if "persistent_kernel" not in decoding_cfg and decoding_cfg.get(
            "force_per_step"
        ):
            persistent = False
        if isinstance(persistent, str):
            persistent = {"true": True, "false": False}.get(
                persistent.lower(), persistent.lower()
            )
        if not (isinstance(persistent, bool) or persistent == "auto"):
            raise ValueError(
                f"persistent_kernel must be true, false or auto, got {persistent!r}"
            )
        self._persistent = None if persistent == "auto" else persistent
        self._host_check_every = int(knob(decoding_cfg, "host_check_every", 8))
        self._skip_converged = bool(knob(decoding_cfg, "skip_converged", True))
        self._warp_per_check = bool(knob(decoding_cfg, "warp_per_check", True))
        self._fuse_vn = bool(knob(decoding_cfg, "fuse_vn", True))
        self._f64_int = bool(knob(decoding_cfg, "f64_int_compare", True))
        self._vn_gather = bool(knob(decoding_cfg, "vn_gather", True))
        self._force_per_step = (
            self._persistent is False
            or not self._skip_converged
            or not self._warp_per_check
            or not self._fuse_vn
            or not self._vn_gather
        )
        self._sm_count = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        with torch.cuda.device(self.device):
            self._max_coresident = self._ext.persistent_max_blocks(
                torch.empty(0, dtype=self.dtype, device=self.device),
                PERSISTENT_THREADS,
                False,
                self._padded,
                self._f64_int,
            )
        logger.info("bp_norm_min_sum_cuda decoder ready.")

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """Per-step path hook, called at the end of iteration i (after the
        convergence update and the early-exit checks). With self._hook_before_stop
        True, it runs before those checks, including on the iteration that breaks.
        self._hook_final is True when those checks would end the loop.
        l_v [B, N_ext] is the posterior LLR (column N is
        the +inf dummy), e_v [B, N_ext] uint8 the hard decision, active [B] bool
        the samples still unconverged, syndrome [B, M] the decoder-dtype
        syndrome. The hook may change l_v in place on active rows only; the next
        iteration reads it, and an active row's l_v at loop end is the returned
        llr. Converged rows, e_v and syndrome are read-only, and the hook cannot
        mark rows converged. It may set self._hook_stop to a [B] bool tensor:
        True active rows stop with their current l_v/e_v and iteration i, keeping
        converge == 0. The mask is reset to None before each call and forward.
        A subclass that overrides it always runs the per-step path.
        The default does nothing."""

    def _exit_hook(self, l_v, e_v, num_iters, converges) -> None:
        """Called once after the decode loop on both paths, with num_iters final
        (no -1 left), before the dummy column is stripped. It may change l_v,
        e_v, num_iters and converges in place; they are the returned outputs.
        It runs before cap.observe(num_iters), so during the rebatch cap warm-up
        a hook that rewrites num_iters changes the iteration histogram the cap
        is chosen from. The default does nothing."""

    def _kernel_knobs(self) -> tuple:
        """(warp_per_check, fuse_vn, f64_int_compare, padded, vn_gather) for forward: the
        knob values on the plain extension, the defaults on a wrapped one (the
        quantized subclasses, whose fixed-point kernels exist for the defaults
        only)."""
        if self._ext is not _EXT:
            return True, True, True, False, True
        return (
            self._warp_per_check,
            self._fuse_vn,
            self._f64_int,
            self._padded,
            self._vn_gather,
        )

    def _use_persistent(self, B: int, capped: bool) -> bool:
        """True if a batch of B samples runs on the persistent kernel: small codes
        (M <= 120) or batches of at least 4 samples per SM on codes with
        M <= PERSISTENT_MAX_M, or any B with persistent_kernel true; never with
        persistent_kernel false (or force_per_step), skip_converged false,
        warp_per_check false, fuse_vn false, vn_gather false or an overridden
        _iter_hook, never when no persistent block fits on an SM, and with the cap on only when all B
        blocks are co-resident."""
        if self._force_per_step or type(self)._iter_hook is not create._iter_hook:
            return False
        if self._max_coresident == 0:
            return False
        if capped and B > self._max_coresident:
            return False
        if self._persistent:
            return True
        M = self.H_shape[0]
        return M <= 120 or (B >= 4 * self._sm_count and M <= PERSISTENT_MAX_M)

    def forward(self, io_dict: dict) -> dict:
        """
        Run BP Normalized Min-Sum decoding on the CSR edge layout (row_ptr, col,
        VN_eid; b_c2v is [B, nnz + 1] with the last slot always zero).

        Two paths give the same outputs bit for bit; _use_persistent picks one
        per call from the batch size B:

        - persistent: one launch of bp_nms_persistent runs the whole loop with
          one block of PERSISTENT_THREADS threads per sample, each block stopping
          when its sample converges. With the rebatch cap active the kernel
          stops every running sample after the first iteration at which at
          least ceil(frac * B) samples have converged, the same rule as the
          per-step host check below.
        - per-step: each iteration launches vn_cn_update_csr (computes the
          VN→CN message l_v − b_c2v on the fly and overwrites b_c2v in place;
          iteration 1 reads u_init with b_c2v = 0), then llr_hard_update_csr
          (l_v and a uint8 hard decision e_v), syndrome_check_csr (sets a
          per-sample mismatch flag) and convergence_flag_update. All three
          per-sample kernels skip samples that already converged, so their l_v,
          e_v and b_c2v stay at the convergence iteration and are returned
          directly (with skip_converged false the kernels run every sample and
          the converged samples' l_v and e_v are copied out at their
          convergence iteration). The host-side early-exit check runs every
          host_check_every (default 8, 0 never) iterations when the rebatch cap
          is inactive, and every iteration when the cap is active. A
          subclass's _iter_hook runs at the end of each iteration that does not
          break the loop, or before early-exit checks with _hook_before_stop.

        With edge_layout padded both paths run on col_pad and VN_eid_pad, and
        b_c2v is [B, M * D + 1]. On the per-step path, fuse_vn false fills an
        a_v2c buffer shaped like b_c2v with vn_update_csr before each check
        update, warp_per_check false and f64_int_compare false pick the serial and the
        floating-point check update, and vn_gather false replaces
        llr_hard_update_csr with llr_hard_update_atomic.

        Both paths then call _exit_hook once.

        Parameters (from io_dict)
        -------------------------
        synd  : [B, M]  binary syndrome tensor
        llr0  : [B, N]  initial channel LLR (log P(0)/P(1) per qubit)

        Updates io_dict with
        --------------------
        e_v      : [B, N]   hard-decision error estimate
        llr      : [B, N]   final per-variable LLR
        iter     : [B]      iteration at which each sample converged (max_iter,
                            or the cap's stop iteration, if not)
        converge : [B]      1 if converged, 0 otherwise
        defer    : [B]      bool, True on unconverged samples when the cap
                            stopped the batch before max_iter
        """
        dev = self.device
        # .contiguous() is load-bearing: every kernel indexes raw data pointers
        # assuming row-major layout. A transposed-stride syndrome (e.g. built
        # via (H @ e.T).T) would otherwise be read as other samples' bits.
        syndrome = io_dict["synd"].to(dtype=self.dtype, device=dev).contiguous()
        B, M = syndrome.shape
        self.batch_size = B
        self._hook_stop = None
        self._hook_rows = None
        self._hook_final = False

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

        num_iters = torch.zeros(B, dtype=torch.int64, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        cap_applied = bool(
            self.cap is not None
            and self.cap.done
            and self.cap.frac is not None
            and not self.cap_bypass
        )
        cap_stopped = False
        warp, fuse_vn, f64_int, padded, vn_gather = self._kernel_knobs()
        if padded:
            col, vn_eid = self.col_pad, self.VN_eid_pad
        else:
            col, vn_eid = self.col, self.VN_eid
        b_c2v = torch.zeros(B, col.numel() + 1, dtype=self.dtype, device=dev)
        # Iteration 1 writes every entry of l_v and e_v (no sample has
        # converged yet), so they start uninitialised.
        l_v = torch.empty(B, self.N_ext, dtype=self.dtype, device=dev)
        e_v = torch.empty(B, self.N_ext, dtype=torch.uint8, device=dev)

        if self._use_persistent(B, cap_applied):
            cnt, stop_count = None, 0
            if cap_applied:
                cnt = torch.zeros(
                    2 * (self.max_iter + 1), dtype=torch.int32, device=dev
                )
                stop_count = int(math.ceil(self.cap.frac * B))
            self._ext.bp_nms_persistent(
                u_init,
                syndrome_neg_bc,
                syndrome,
                self.row_ptr,
                col,
                vn_eid,
                b_c2v,
                l_v,
                e_v,
                num_iters,
                converges,
                self.N,
                self.max_iter,
                PERSISTENT_THREADS,
                cnt,
                stop_count,
                **({} if f64_int else {"f64_int": False}),
            )
            if cap_applied:
                # the kernel writes the stop iteration to every unconverged row
                cap_stopped = bool(
                    ((converges == 0) & (num_iters < self.max_iter)).any()
                )
        else:
            every = self._host_check_every
            cap_frac = self.cap.frac if cap_applied else None
            mismatch = torch.zeros(B, dtype=torch.int32, device=dev)
            num_iters.fill_(-1)
            hooked = type(self)._iter_hook is not create._iter_hook
            hook_before_stop = hooked and getattr(self, "_hook_before_stop", False)
            if hooked:
                self._hook_rows = torch.arange(B, device=dev)
            # num_iters the three compute kernels skip on: all -1 runs every sample
            run = num_iters
            if not self._skip_converged:
                run = torch.full_like(num_iters, -1)
                l_snap = torch.empty_like(l_v)
                e_snap = torch.empty_like(e_v)
            # keyword arguments only off their defaults: the quantized
            # subclasses' wrapped extension takes positional arguments only
            cn_kw = {
                k: False for k, on in (("warp", warp), ("f64_int", f64_int)) if not on
            }
            if not fuse_vn:
                a_v2c = torch.empty_like(b_c2v)
                cn_kw["from_lv"] = False
            if not vn_gather:
                vn_sum = torch.empty_like(l_v)

            for i in range(1, self.max_iter + 1):
                beta = 1.0 - 2.0 ** (-i)
                l_in = u_init if i == 1 else l_v
                if not fuse_vn:
                    self._ext.vn_update_csr(l_in, b_c2v, col, a_v2c, run, self.N)
                    l_in = a_v2c
                self._ext.vn_cn_update_csr(
                    l_in,
                    syndrome_neg_bc,
                    self.row_ptr,
                    col,
                    b_c2v,
                    run,
                    beta,
                    self.N,
                    **cn_kw,
                )
                if not vn_gather:
                    self._ext.llr_hard_update_atomic(
                        u_init, b_c2v, col, vn_sum, l_v, e_v, run
                    )
                else:
                    self._ext.llr_hard_update_csr(u_init, b_c2v, vn_eid, l_v, e_v, run)
                self._ext.syndrome_check_csr(
                    e_v, self.row_ptr, col, syndrome, run, mismatch, self.N
                )
                self._ext.convergence_flag_update(mismatch, num_iters, converges, i)
                if not self._skip_converged:
                    new = (num_iters == i).unsqueeze(1)
                    torch.where(new, l_v, l_snap, out=l_snap)
                    torch.where(new, e_v, e_snap, out=e_snap)
                should_break = False
                if cap_frac is not None:
                    # cap active: count converged each iteration (host sync) and stop
                    # once the learned fraction is reached or every sample converges.
                    n_conv = int(converges.sum())
                    if n_conv >= cap_frac * B or n_conv == B:
                        cap_stopped = n_conv < B and i < self.max_iter
                        should_break = True
                elif (
                    every
                    and i % every == 0
                    and not (num_iters == -1).any().item()
                ):
                    should_break = True
                if hooked and (hook_before_stop or not should_break):
                    self._hook_stop = None
                    self._hook_final = should_break
                    active = num_iters == -1
                    self._iter_hook(i, l_v, e_v, active, syndrome)
                    if self._hook_stop is not None:
                        stop = self._hook_stop & active
                        num_iters.masked_fill_(stop, i)
                        if not self._skip_converged:
                            run.masked_fill_(stop, i)
                            l_snap[stop] = l_v[stop]
                            e_snap[stop] = e_v[stop]
                        if not hook_before_stop:
                            should_break |= not (num_iters == -1).any().item()
                if hook_before_stop:
                    should_break = not (num_iters == -1).any().item()
                    if cap_frac is not None:
                        n_conv = int(converges.sum())
                        cap_stopped = (
                            n_conv >= cap_frac * B and n_conv < B and i < self.max_iter
                        )
                        should_break |= n_conv >= cap_frac * B
                if should_break:
                    break

            if not self._skip_converged:
                done = (num_iters != -1).unsqueeze(1)
                l_v = torch.where(done, l_snap, l_v)
                e_v = torch.where(done, e_snap, e_v)
            num_iters.masked_fill_(num_iters == -1, i)

        self._exit_hook(l_v, e_v, num_iters, converges)
        self.cap_active_last = cap_stopped
        # rows main drops and decodes again: the unconverged rows of a batch the
        # cap stopped before max_iter
        defer = (
            converges.flatten() == 0
            if cap_stopped
            else torch.zeros(B, dtype=torch.bool, device=dev)
        )

        # Free the edge buffer before the hard decision is widened to the
        # decoder dtype, so the widening does not raise the peak.
        del b_c2v, u_init
        e_out = e_v.to(self.dtype)
        l_out = l_v

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
                "defer": defer,
            }
        )
        return io_dict
