"""union_find_cuda.py -- standalone CUDA Union-Find decoder (v1).

A from-scratch GPU Union-Find decoder built like ``osd_0_cuda`` / ``bp_norm_min_sum_cuda``:
a self-contained ``nn.Module`` whose whole decode runs on the GPU via custom kernels
(``cuda/union_find_kernel.cu``). It subclasses nothing from ``union_find.py`` and uses
nothing from ``mwpm.py`` -- no robin-hood hash-map order, no per-thread arena, no blossom,
no CPU / PyTorch fallback.

It reproduces the batched-tensor decoder in ``union_find.py``'s ``forward()`` bit-for-bit
(``e-diff == 0``): the graphlike Delfosse-Nickerson pipeline grow/fuse -> spanning forest ->
leaf-rake peel over the detector graph (V = M+1 vertices with a boundary vertex at id 0,
N edges = qubit columns). Every tie-break is min-vertex-id / min-edge-index, so the kernel is
general **graphlike** (toric *and* surface / open-boundary), matching ``union_find.py``'s
domain and, transitively, its qsurface logical-error-rate match.

Two execution paths, mirroring the two reference files:

  * FUSED (default): one kernel launch, one thread block per shot, every working array in
    shared memory. Used whenever the per-shot working set fits the shared-memory limit.

  * PER-STEP (fallback / ``force_per_step: true``): a Python loop over the three modular phase
    kernels (uf_grow_fuse -> uf_spanning_forest -> uf_peel) in global memory -- for codes too
    large for shared memory and for debugging single phases. Bit-identical to the fused path
    (both call the same device functions).

CUDA-only. For the qsurface-algorithm port validated by logical error rate (not bit-exact to
``union_find.py``), see the sibling ``union_find_cuda_v2.py``.
"""

import os
import math

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


# ── Extension loader (mirrors osd_0_cuda / bp_norm_min_sum_cuda) ────────────────

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
    """JIT-compile union_find_kernel.cu on first call; cache thereafter."""
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    handles = _dll_dirs()
    decoder_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(decoder_dir, "cuda", "union_find_kernel.cu")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    logger.info(
        "Compiling union_find_cuda kernel -- first use in each Python environment "
        "runs nvcc (a few minutes, no progress output); later runs load from cache."
    )
    _EXT = load(
        name="union_find_kernel",
        sources=[src],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    for h in handles:
        try:
            h.close()
        except Exception:
            pass
    logger.info("union_find_cuda kernel compiled.")
    return _EXT


# ── Host graph build (standalone replica of union_find.py's graphlike build) ─────


def _edge_endpoints_from_parity(H_np):
    """The (up to two) detector vertices each qubit column links, as graphlike edges.

    Replicates ``union_find.py._edge_endpoints_from_parity`` exactly (kept local so this
    module depends on nothing in ``union_find.py``). A boundary vertex is prepended at id 0
    and the M detector rows shift to 1..M, so the internal vertex count is ``V = M + 1``:

      * weight 2  -> ``(r0+1, r1+1)``   interior edge between two detectors
      * weight 1  -> ``(0, r0+1)``      boundary edge (open boundary, e.g. surface code)
      * weight 0  -> ``(0, 0)``         boundary self-loop (inert)
      * weight >2 -> rejected           not graphlike

    Returns ``(eu, ev, V, is_pure_toric)``: int64 arrays [N] of (min, max) endpoints, the
    vertex count ``V = M + 1``, and whether every column has weight exactly two.
    """
    H_np = np.asarray(H_np).astype(np.uint8)
    M, N = H_np.shape
    # np.nonzero on H^T visits columns 0..N-1 in order, rows increasing within a column.
    cols, rows = np.nonzero(H_np.T)
    counts = np.bincount(cols, minlength=N)
    if np.any(counts > 2):
        bad = int(np.argmax(counts > 2))
        raise ValueError(
            f"Column {bad} has weight {int(counts[bad])} > 2; H is not graphlike, so the "
            f"union_find decoder does not apply (use a hypergraph/correlated decoder)."
        )
    eu = np.zeros(N, dtype=np.int64)  # weight-0 default: boundary self-loop (0, 0)
    ev = np.zeros(N, dtype=np.int64)
    per_col = np.split(
        rows + 1, np.cumsum(counts)[:-1]
    )  # detectors per column, +1 shift
    for q in range(N):
        r = per_col[q]
        if r.size == 2:
            eu[q], ev[q] = int(r[0]), int(r[1])  # rows increasing => already (min, max)
        elif r.size == 1:
            ev[q] = int(r[0])  # boundary edge (0, detector); eu stays 0
    is_pure_toric = bool(np.all(counts == 2))
    return eu, ev, M + 1, is_pure_toric


# ── Decoder class ───────────────────────────────────────────────────────────────


class create(nn.Module):
    """
    Standalone CUDA Union-Find decoder, bit-exact to union_find.py's forward().

    Accepted YAML keys (under ``decoder:``):
        check_type  : hx | hz          (default: hx)
        dtype       : float32 | float64 | float16 | bfloat16  (default: float64)
        device:
          device_type : cuda           (only cuda is supported)
          device_idx  : int            (default: 0)
        force_per_step : bool          (optional; selects the modular per-step path)
    """

    def __init__(self, decoder_cfg: dict, **kwargs) -> None:
        super().__init__()
        logger.info("Creating union_find_cuda decoder.")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "union_find_cuda requires a CUDA-capable GPU. Use union_find for CPU."
            )

        # ── device ────────────────────────────────────────────────────────────
        device_cfg = decoder_cfg.get("device", {})
        device_type = device_cfg.get("device_type", "cuda")
        if device_type != "cuda":
            logger.warning(
                f"union_find_cuda only supports cuda; ignoring "
                f"device_type='{device_type}'."
            )
        device_idx = device_cfg.get("device_idx", 0)
        if device_idx >= torch.cuda.device_count():
            logger.warning(f"device_idx={device_idx} exceeds available GPUs; using 0.")
            device_idx = 0
        self.device = torch.device(f"cuda:{device_idx}")

        # ── dtype ─────────────────────────────────────────────────────────────
        dtype_str = decoder_cfg.get("dtype", "float64")
        if dtype_str not in {"float16", "bfloat16", "float32", "float64"}:
            logger.warning(f"Invalid dtype '{dtype_str}'; defaulting to float64.")
            dtype_str = "float64"
        self.dtype = torch.__dict__[dtype_str]

        self.check_type = decoder_cfg.get("check_type", "hx").lower()
        if self.check_type not in {"hx", "hz"}:
            logger.warning(f"Invalid check_type='{self.check_type}'; defaulting to hx.")
            self.check_type = "hx"

        # ── matrix bundle -> graphlike edge endpoints ─────────────────────────
        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "union_find_cuda requires a pre-loaded MatrixBundle via `bundle`."
            )
        H_shape, _, V_c_col, H_matrix = bundle.select(self.check_type)
        self.H_shape = H_shape
        # V_c_col (parity-row -> qubit-column map) is read by the perfect syndrome measurer
        # when this decoder is decoders[0]; expose it like every other decoder.
        self.V_c_col = nn.Parameter(V_c_col.to(self.device), requires_grad=False)
        self.M, self.N = int(H_shape[0]), int(H_shape[1])

        H_np = (H_matrix.detach().cpu().numpy() != 0).astype(np.uint8)
        eu, ev, self.V, self.is_pure_toric = _edge_endpoints_from_parity(H_np)
        self.eu = nn.Parameter(
            torch.from_numpy(eu.astype(np.int32)).to(self.device), requires_grad=False
        )
        self.ev = nn.Parameter(
            torch.from_numpy(ev.astype(np.int32)).to(self.device), requires_grad=False
        )

        # ── metadata expected by the framework ────────────────────────────────
        self.algo = "union_find"
        self.num_max_iter = self.N + 1
        self.batch_size = 1

        # ── load extension + choose path ──────────────────────────────────────
        self._ext = _load_ext()
        smem_needed = self._ext.fused_smem_bytes(self.V, self.N)
        smem_limit = self._ext.fused_smem_limit()
        self._use_fused = smem_needed <= smem_limit and not decoder_cfg.get(
            "force_per_step", False
        )
        if not self._use_fused:
            logger.info(
                f"union_find_cuda using per-step path "
                f"(fused needs {smem_needed} B shared memory, limit {smem_limit} B)."
            )

        # Block size: cover the larger of V vertices / N edges, round to a warp, cap 512.
        span = max(self.V, self.N)
        self._block_size = min(max(32 * math.ceil(span / 32), 64), 512)
        logger.info(
            f"union_find_cuda decoder ready ({self.device}, "
            f"{'fused' if self._use_fused else 'per-step'})."
        )

    # ── Forward pass ────────────────────────────────────────────────────────────

    def forward(self, io_dict: dict) -> dict:
        dev = self.device
        synd_in = io_dict["synd"]
        B, M = synd_in.shape
        self.batch_size = B
        assert M == self.M, f"syndrome width {M} != H rows {self.M}"

        # Internal syndrome [B, V] int32; boundary vertex 0 is never a defect.
        synd_det = (synd_in != 0).to(device=dev, dtype=torch.int32)  # [B, M]
        synd = torch.zeros((B, self.V), dtype=torch.int32, device=dev)
        synd[:, 1:] = synd_det
        synd = synd.contiguous()

        if self._use_fused:
            corr = self._ext.uf_fused(
                synd, self.eu, self.ev, int(self.V), int(self.N), self._block_size
            )
        else:
            corr = self._run_per_step(synd)

        e_v = corr.to(device=dev, dtype=self.dtype)  # [B, N]
        # sign-encoded soft output (+1 bit 0, -1 bit 1); UF carries no real LLR
        llr = (1.0 - 2.0 * e_v).to(device=dev, dtype=self.dtype)
        converge = torch.ones(B, dtype=torch.int64, device=dev)
        # iter mirrors the syndrome weight per shot (clamped >= 1), as in union_find.py
        iters = synd.sum(dim=1).clamp(min=1).to(torch.int64)

        io_dict.update({"e_v": e_v, "iter": iters, "llr": llr, "converge": converge})
        return io_dict

    def _run_per_step(self, synd: torch.Tensor) -> torch.Tensor:
        """Modular-kernel path: Python loop over the three phase kernels in global memory."""
        dev = self.device
        B = synd.shape[0]
        V, N, bs = self.V, self.N, self._block_size

        def _v(dtype=torch.int32):
            return torch.empty((B, V), dtype=dtype, device=dev)

        label = _v()
        support = torch.empty((B, N), dtype=torch.int32, device=dev)
        par = _v()
        dist = _v()
        best = _v()
        parent = _v()
        tree = _v()
        child = _v()
        syndcur = _v()
        active = _v()
        corr = torch.empty((B, N), dtype=torch.int32, device=dev)

        self._ext.uf_grow_fuse(
            synd, label, support, par, self.eu, self.ev, int(V), int(N), bs
        )
        self._ext.uf_spanning_forest(
            label,
            support,
            dist,
            best,
            parent,
            tree,
            self.eu,
            self.ev,
            int(V),
            int(N),
            bs,
        )
        self._ext.uf_peel(
            synd, parent, tree, child, syndcur, par, active, corr, int(V), int(N), bs
        )
        return corr
