import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from syndrilla.decoder.knobs import knob

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
        "Compiling osd_0_cuda kernels: the first use in each Python environment "
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
        extra_include_paths=[cuda_dir],  # so osd0.h resolves
        extra_cuda_cflags=["-O3", "-diag-suppress=221"] + msvc_extra,
        extra_cflags=["/Zc:preprocessor"] if os.name == "nt" else [],
        verbose=True,
    )
    logger.info("osd_0_cuda kernels compiled.")
    return _EXT


def _pack_H(rows: np.ndarray, cols: np.ndarray, M: int, W: int) -> np.ndarray:
    """Bit-pack the {0,1} [M, N] matrix given by its nonzero (rows, cols) into uint64
    [M, W]. Returns the uint64 array."""
    packed = np.zeros((M, W), dtype=np.uint64)
    word = (cols >> 6).astype(np.int64)
    bit = (cols & 63).astype(np.uint64)
    np.bitwise_or.at(packed, (rows, word), (np.uint64(1) << bit))
    return packed


def _gf2_rank(packed: np.ndarray, M: int, N: int) -> int:
    """Rank over GF(2) of the bit-packed [M, W] rows from _pack_H, eliminating over
    columns 0..N-1 (host, slow; last-resort fallback)."""
    rows = packed.copy()
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

    Two phases per sample: a single-kernel scan finds the pivot columns (the
    greedy independent set S of H's columns in reliability order) by eliminating
    only the rows without a pivot; then Gauss-Jordan over the pivot columns gives
    the estimate. Columns without a pivot cause no row operations, so e_v is the
    same as Gauss-Jordan over all N columns.

    Early stop: the scan also reduces the syndrome s by the same row operations
    and stops once it is zero on every row without a pivot. At that point s lies
    in the span of the pivot columns found so far (a prefix of S). H_S has full
    column rank, so H_S e = s has a unique solution, which is then zero on the
    later pivots; Gauss-Jordan over the prefix gives the same e_v. A syndrome
    outside the column space of H never stops early and uses all of S.

    Accepted YAML keys (under ``decoder:``):
        algorithm   : osd_0_cuda
        check_type  : hx | hz          (default: hx)
        dtype       : float32 | float64 | float16 | bfloat16  (default: float64)
        device:
          device_type : cuda           (only cuda is supported)
          device_idx  : int            (default: 0)
        force_per_step  : bool         (optional; selects the modular debug path)
        workspace_bytes : int          (optional; GPU workspace cap per chunk, default 4 GiB)
        osd_early_stop  : bool         (default true; false passes no syndrome to the
                                        scan, so it runs to rank(H); turns
                                        osd_prefix_scan off)
        osd_prefix_scan : bool         (default true; false sorts the whole order once)
        osd_solve_by_pivots : bool     (default true; false eliminates over rank(H)
                                        columns instead of the largest pivot count)
        osd_skip_converged : bool      (default true; false decodes every sample and
                                        keeps the result only where converge is 0)
        osd_column_scan : bool         (default true; false skips the scan and
                                        eliminates over all N columns of the order)
        pruning_opt     : bool         (default true; false makes the defaults of
                                        osd_early_stop, osd_prefix_scan,
                                        osd_solve_by_pivots and osd_skip_converged
                                        false)
        memory_opt      : bool         (default true; false makes the defaults of
                                        osd_column_scan false and of workspace_bytes
                                        1 << 60)
        fusion_opt, mapping_opt, gather_opt : no OSD members, ignored
    """

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        super().__init__()
        logger.info("Creating osd_0_cuda decoder.")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "osd_0_cuda requires a CUDA-capable GPU. Use osd_0 for CPU execution."
            )

        device_cfg = decoding_cfg.get("device", {})
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

        dtype_str = decoding_cfg.get("dtype", "float64")
        if dtype_str not in {"float16", "bfloat16", "float32", "float64"}:
            logger.warning(f"Invalid dtype '{dtype_str}'; defaulting to float64.")
            dtype_str = "float64"
        self.dtype = torch.__dict__[dtype_str]

        self.check_type = decoding_cfg.get("check_type", "hx").lower()
        if self.check_type not in {"hx", "hz"}:
            logger.warning(f"Invalid check_type='{self.check_type}'; defaulting to hx.")
            self.check_type = "hx"

        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "osd_0_cuda requires a pre-loaded MatrixBundle via the `bundle` kwarg."
            )
        H_shape, _, V_c_col, _ = bundle.select(self.check_type)
        self.H_shape = H_shape
        self.V_c_col = nn.Parameter(V_c_col.to(self.device), requires_grad=False)
        self.M, self.N = int(H_shape[0]), int(H_shape[1])
        self.force_per_step = bool(decoding_cfg.get("force_per_step", False))
        self.workspace_bytes = int(knob(decoding_cfg, "workspace_bytes", 4 << 30))
        self.osd_early_stop = bool(knob(decoding_cfg, "osd_early_stop", True))
        self.osd_prefix_scan = (
            bool(knob(decoding_cfg, "osd_prefix_scan", True)) and self.osd_early_stop
        )
        self.osd_solve_by_pivots = bool(knob(decoding_cfg, "osd_solve_by_pivots", True))
        self.osd_skip_converged = bool(knob(decoding_cfg, "osd_skip_converged", True))
        self.osd_column_scan = bool(knob(decoding_cfg, "osd_column_scan", True))
        # Rows of T the pivot scan stores per sample (Tr, TuT); see _scan.
        self.pool_rows = min(self.M, max(64, (self.M + 5) // 6))
        self.pool_cols = min(self.M, max(64, (self.M + 2) // 3))
        # Columns of the reliability order the scan tries first, see forward;
        # the whole order without osd_prefix_scan or osd_column_scan.
        self.prefix = (
            self.N // 16
            if self.N >= 1 << 14 and self.osd_prefix_scan and self.osd_column_scan
            else self.N
        )
        self.last_pivot_pos = (
            -1
        )  # highest pivot order position over the last forward call
        # Per-sample scan statistics of the last forward call: columns scanned
        # ("end"), pivots found ("pivots"), and whether the scan stopped early.
        self.scan_stats = None

        # CSC of H for the pivot scan and the column gather.
        V_c_col_np = V_c_col.detach().cpu().numpy()
        rows, pos = np.nonzero(V_c_col_np != self.N)
        cols = V_c_col_np[rows, pos].astype(np.int64)
        by_col = np.lexsort((rows, cols))
        colptr = np.zeros(self.N + 2, dtype=np.int64)  # column N: empty, for padding
        np.cumsum(np.bincount(cols, minlength=self.N), out=colptr[1 : self.N + 1])
        colptr[self.N + 1] = colptr[self.N]
        self.register_buffer(
            "colptr", torch.from_numpy(colptr).to(self.device), persistent=False
        )
        self.register_buffer(
            "rowidx",
            torch.from_numpy(rows[by_col].astype(np.int32)).to(self.device),
            persistent=False,
        )

        self.algo = "osd_0"
        self.num_max_iter = self.N  # matches osd_0.py:122
        self.batch_size = 1

        self._ext = _load_ext()
        self._smem_limit = self._ext.fused_smem_limit()
        # Block size for the fused kernel: cover M rows, round to a warp, cap 512.
        self._block_size = min(max(32 * math.ceil(self.M / 32), 64), 512)

        t = time.time()
        self.A_rank, src = self._rank(rows, cols)
        logger.info(f"rank(H) = {self.A_rank} from {src} in {time.time() - t:.2f} s.")
        logger.info("osd_0_cuda decoder ready.")

    def _rank(self, rows, cols):
        """GF(2) rank of H: ldpc if it imports, else one GPU pivot scan,
        else the host loop."""
        try:
            import ldpc.mod2
            import scipy.sparse

            H = scipy.sparse.csr_matrix(
                (np.ones(len(rows), np.uint8), (rows, cols)), shape=(self.M, self.N)
            )
            return int(ldpc.mod2.rank(H, method="sparse")), "ldpc.mod2.rank"
        except (ImportError, TypeError):  # no ldpc, or an ldpc without method=
            pass
        try:
            order = torch.arange(self.N, dtype=torch.int32, device=self.device)[None]
            _, found, _, _ = self._scan(order, self.M)
            return int(found[0]), "GPU scan"
        except RuntimeError as err:  # e.g. out of GPU memory
            logger.warning(f"GPU rank failed ({err}); using the slow host loop.")
        W = (self.N + 63) >> 6
        return _gf2_rank(_pack_H(rows, cols, self.M, W), self.M, self.N), "host loop"

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

        # rows marked in io_dict["defer"] are re-decoded by the batch loop, OSD skips them
        defer = (
            io_dict["defer"].to(device=converge.device, dtype=torch.bool)
            if "defer" in io_dict
            else None
        )
        nc = converge == 0 if defer is None else (converge == 0) & ~defer
        idx = nc.nonzero(as_tuple=True)[0].to(dev)
        # samples OSD decodes: all non-deferred ones without osd_skip_converged
        if self.osd_skip_converged:
            run = idx
        elif defer is None:
            run = torch.arange(B, device=dev)
        else:
            run = (~defer).nonzero(as_tuple=True)[0].to(dev)
        if run.numel() > 0:
            llr_sub = io_dict["llr"].to(dtype=self.dtype, device=dev)[run]
            synd_sub = io_dict["synd"].to(device=dev)[run]

            # Column order: most reliable handling matches osd_0.py:140
            # (stable ascending sort of the posterior LLR). The scan first runs
            # on the first k = `prefix` columns of that order. A sample that
            # does not stop inside them is solved again on the first 4k columns
            # (skipped when 4k >= N), and a sample that does not stop there
            # either is solved with the full order.
            synd_u8 = synd_sub.to(torch.uint8).contiguous()
            order = self._order_prefix(llr_sub, self.prefix)
            e_sub = self._solve(synd_u8, order)
            prev = order.shape[1]
            if prev < self.N:
                stats, last = self.scan_stats, self.last_pivot_pos
                stages = [(prev, order.shape[0])]  # (order width, samples solved)
                for width in ([4 * prev] if 4 * prev < self.N else []) + [self.N]:
                    redo = (~stats["stopped"]).nonzero(as_tuple=True)[0]
                    if not redo.numel():
                        break
                    stages.append((width, redo.numel()))
                    sub = self._order_prefix(llr_sub[redo], width)
                    e_sub[redo] = self._solve(synd_u8[redo], sub)
                    for key, val in self.scan_stats.items():
                        stats[key][redo] = val
                    last = max(last, self.last_pivot_pos)
                self.scan_stats, self.last_pivot_pos = stats, last
                logger.debug("osd_0_cuda: (order width, samples solved): {}", stages)
            e_v[idx] = e_sub if self.osd_skip_converged else e_sub[nc.to(dev)[run]]
            iter_out = iter_out.to(dev)
            iter_out[idx] = self.num_max_iter

        io_dict.update(
            {
                "e_v": e_v,
                "iter": iter_out,
                "converge": torch.ones_like(converge)
                if defer is None
                else torch.where(defer, converge, torch.ones_like(converge)),
            }
        )
        return io_dict

    def _solve(self, synd_u8, order):
        """OSD-0 for every sample, in chunks that keep the GPU workspace under
        workspace_bytes: scan chunks sized by _scan_bytes, and elimination
        batches sized by the chunk's largest pivot count K (rank(H) without
        osd_solve_by_pivots). Without osd_column_scan the chunks skip the scan and
        eliminate over all columns of order, and scan_stats is None."""
        B, M, N, rank = order.shape[0], self.M, self.N, self.A_rank
        e = torch.zeros(B, N, dtype=torch.uint8, device=self.device)
        self.last_pivot_pos = -1
        per_sample = self._scan_bytes(self.pool_rows, self.pool_cols, rank)
        chunk = max(1, self.workspace_bytes // per_sample)
        stats = []
        for i in range(0, B, chunk):
            o, s = order[i : i + chunk], synd_u8[i : i + chunk]
            if self.osd_column_scan:
                # without osd_early_stop the scan gets no syndrome and never stops
                piv_pos, found, stopped, end = self._scan(
                    o, rank, s if self.osd_early_stop else None
                )
                if o.shape[1] == N and bool(((found != rank) & ~stopped).any()):
                    raise RuntimeError("osd_0_cuda: pivot count differs from rank(H).")
                stats.append((end, found, stopped, self.last_overflow))
                K = int(found.max())
                if K:
                    last = piv_pos.gather(1, (found - 1).clamp(min=0).long()[:, None])
                    self.last_pivot_pos = max(
                        self.last_pivot_pos, int(last[found > 0].max())
                    )
                if not self.osd_solve_by_pivots:
                    K = rank
                # Pivot columns in order; past a sample's own pivot count, column N
                # (empty) pads the row to K.
                valid = torch.arange(K, device=o.device)[None] < found[:, None]
                cols = torch.where(valid, o.gather(1, piv_pos[:, :K].long()), N)
                cols = cols.to(torch.int32).contiguous()
                del piv_pos, valid
            else:
                cols, K = o, o.shape[1]
            # Elimination bytes per sample: ws [M, Wk] int64 plus 16M (the
            # shifted int64 syndrome, or row_pcol, colw and the pivot buffers).
            elim = M * (((K + 64) >> 6) * 8 + 16)
            sub = max(1, (self.workspace_bytes - cols.numel() * 4) // elim)
            for a in range(0, cols.shape[0], sub):
                c = cols[a : a + sub]
                ws, row_pcol = self._eliminate(s[a : a + sub], c, rank)
                self._ext.osd_solve(ws, row_pcol, c, e[i + a : i + a + sub], K)
                del ws, row_pcol
            del cols
        if not stats:
            self.scan_stats = None
            return e
        end, found, stopped, ovf = (torch.cat(x) for x in zip(*stats))
        self.scan_stats = {
            "end": end,
            "pivots": found,
            "stopped": stopped,
            "overflow": ovf,
        }
        # lazy=True: the host syncs in the lambdas run only if a sink takes DEBUG.
        logger.opt(lazy=True).debug(
            "osd_0_cuda scan: {}/{} samples stopped early; {} rescanned with the "
            "full T after a pool overflow; "
            "mean columns scanned {:.0f} of {}, mean pivots {:.0f} of {}.",
            lambda: int(stopped.sum()),
            lambda: B,
            lambda: int(ovf.sum()),
            lambda: end.double().mean().item(),
            lambda: N,
            lambda: found.double().mean().item(),
            lambda: rank,
        )
        return e

    @staticmethod
    def _order_prefix(llr, k):
        """First k columns ([B, k] int32) of the stable ascending sort of llr:
        every index with llr < t plus the lowest indices with llr == t, where t
        is the k-th smallest value, ordered by (value, index)."""
        if k >= llr.shape[1] or bool(llr.isnan().any()):
            _, order = torch.sort(llr, dim=1, descending=False, stable=True)
            return order[:, :k].to(torch.int32).contiguous()
        t = torch.topk(llr, k, dim=1, largest=False, sorted=False).values
        t = t.max(dim=1, keepdim=True).values
        lt, eq = llr < t, llr == t
        need = k - lt.sum(dim=1, keepdim=True, dtype=torch.int32)
        take = lt | (eq & (eq.cumsum(dim=1, dtype=torch.int32) <= need))
        cols = take.nonzero()[:, 1].view(llr.shape[0], k)  # ascending per row
        _, pos = torch.sort(llr.gather(1, cols), dim=1, stable=True)
        return cols.gather(1, pos).to(torch.int32).contiguous()

    def _scan_bytes(self, rows, cols, rank):
        """GPU bytes per sample of a pivot scan with pools of rows / cols T rows:
        the Tr and TuT pools, the syndrome bits [Uw, 64] and packed syndrome
        [Uw] (int64), the two row maps [M] and piv_pos [rank] (int32), and the
        found / stopped / end / overflow outputs."""
        Uw = (self.M + 63) >> 6
        return (rows + cols + 65) * Uw * 8 + (2 * self.M + rank) * 4 + 12

    def _scan(self, order, rank, synd_u8=None):
        """Pivot-column scan with compact T: pools of pool_rows Tr rows and
        pool_cols TuT rows per sample. A sample that overflows a pool is scanned
        again with pools of M rows (the full T), in batches that fit
        workspace_bytes. Returns the order positions of each sample's pivot
        columns ([B, rank] int32, valid up to the pivot count), the pivot count,
        whether the scan stopped early (only with synd_u8), and the number of
        columns scanned ([B] each). last_overflow ([B] bool) marks the samples
        that overflowed."""
        dev, M = self.device, self.M
        B, Uw = order.shape[0], (M + 63) >> 6
        if synd_u8 is None:
            packed = torch.empty(0, dtype=torch.int64, device=dev)
        else:  # bit r of word r // 64; distinct bits, so the sum is the OR
            bits = torch.zeros(B, Uw, 64, dtype=torch.int64, device=dev)
            bits.view(B, -1)[:, :M] = synd_u8
            bits <<= torch.arange(64, device=dev)
            packed = bits.sum(2)
        out = self._scan_pools(order, rank, packed, self.pool_rows, self.pool_cols)
        ovf = out[4].bool()
        idx = ovf.nonzero(as_tuple=True)[0]
        step = max(1, self.workspace_bytes // self._scan_bytes(M, M, rank))
        for a in range(0, idx.numel(), step):
            ix = idx[a : a + step]
            pk = packed[ix].contiguous() if packed.numel() else packed
            redo = self._scan_pools(order[ix].contiguous(), rank, pk, M, M)
            for t, r in zip(out[:4], redo):
                t[ix] = r
        piv_pos, found, stopped, end, _ = out
        self.last_overflow = ovf
        return piv_pos, found, stopped.bool(), end

    def _scan_pools(self, order, rank, packed, rows, cols):
        """One scan launch with pools of rows Tr / cols TuT rows per sample.
        Returns piv_pos, found, stopped, end and overflow (uint8)."""
        dev, M = self.device, self.M
        B, Uw = order.shape[0], (M + 63) >> 6
        piv_pos = torch.zeros(B, rank, dtype=torch.int32, device=dev)
        found = torch.zeros(B, dtype=torch.int32, device=dev)
        stopped = torch.zeros(B, dtype=torch.uint8, device=dev)
        end = torch.zeros(B, dtype=torch.int32, device=dev)
        ovf = torch.zeros(B, dtype=torch.uint8, device=dev)
        self._ext.osd_scan(
            torch.zeros(B, rows, Uw, dtype=torch.int64, device=dev),
            torch.zeros(B, cols, Uw, dtype=torch.int64, device=dev),
            torch.full((B, M), -1, dtype=torch.int32, device=dev),
            torch.full((B, M), -1, dtype=torch.int32, device=dev),
            self.colptr,
            self.rowidx,
            order,
            piv_pos,
            found,
            packed,
            stopped,
            end,
            ovf,
        )
        return piv_pos, found, stopped, end, ovf

    def _eliminate(self, synd_u8, cols, rank):
        """Gauss-Jordan over the K = cols.shape[1] columns cols[b] (bit K holds
        the syndrome). Returns (ws [B, M, Wk], row_pcol [B, M]: the column index
        into cols[b] of each pivot row, or -1)."""
        dev, M = self.device, self.M
        B, K = cols.shape
        Wk = (K + 64) >> 6
        ws = torch.zeros(B, M, Wk, dtype=torch.int64, device=dev)
        self._ext.osd_gather(ws, self.colptr, self.rowidx, cols, K)
        ws[:, :, K >> 6] |= synd_u8.to(torch.int64) << (K & 63)
        row_pcol = torch.full((B, M), -1, dtype=torch.int32, device=dev)
        found = torch.zeros(B, dtype=torch.int32, device=dev)
        if (
            not self.force_per_step
            and self._ext.fused_smem_bytes(M, Wk) <= self._smem_limit
        ):
            self._ext.osd0_fused(ws, row_pcol, found, K, rank, self._block_size)
            return ws, row_pcol
        colw = torch.empty(B, M, dtype=torch.int64, device=dev)
        pivbuf = torch.full((3, B), 2**31 - 1, dtype=torch.int32, device=dev)
        if K > 0:
            self._ext.osd_load_col(ws, colw, row_pcol, pivbuf, 0)
        for j in range(K):
            self._ext.osd_step(ws, colw, row_pcol, pivbuf, found, j, K)
        return ws, row_pcol
