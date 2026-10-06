import os

import numpy as np
import torch
from loguru import logger

from syndrilla.decoder.mwpm_gpu.mwpm_gpu import (
    INF, NativeMatcher, _corr_from_match_edges, _mp_worker_reconstruct, _parse_H,
    create as _MwpmPy,
)

_KERNEL = None  # compiled once per process

# Observable-mask width, in 64-bit words. MUST equal OBSW in cuda/mwpm_gpu_kernel.cu. The kernel
# packs each shot's correction into OBSW words, so the GPU obs_mask path supports N (qubits ==
# H columns) up to 64*_OBSW. 4 -> N<=256 (surface d<=11); above that the host falls back to CPU.
# CUDA path is known buggy for large H: above 256 columns the kernel is unused and the whole
# batch runs the serial single-process CPU matcher with the pool unused. Uniform weights compile
# the kernel before the N check (~28s wasted); this fallback still logs 'CUDA blossom kernel' ready.
_OBSW = 4


def _load_kernel():
    """JIT-compile mwpm_gpu_kernel.cu on first call; cache thereafter."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from torch.utils.cpp_extension import load

    decoder_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_src = os.path.join(decoder_dir, "cuda", "mwpm_gpu_kernel.cu")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    logger.info(
        "Compiling mwpm_gpu_cuda kernel -- first use in each Python environment runs nvcc "
        "(a few minutes, no progress output); later runs load from cache."
    )
    _KERNEL = load(
        name="mwpm_gpu_cuda_kernel",
        sources=[kernel_src],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    logger.info("mwpm_gpu_cuda kernel compiled.")
    return _KERNEL


def _build_csr(matcher):
    """CSR of the detector graph in ``mwpm_gpu.py``'s exact neighbor order (_build_nodes,
    mwpm_gpu.py:1470-1487): per node, boundary edges first (neighbor = -1), then normal
    edges sorted by qubit/column id. This ordering *is* the tie-break, so it must match
    mwpm_gpu.py byte-for-byte.

    Each edge's observable is qubit ``j`` packed into an ``_OBSW``-word bitmask: word
    ``j // 64`` gets bit ``j % 64``. This matches the kernel's OBSW-word ``Obs`` layout and
    lifts the old single-word N<=64 cap (the kernel only ever XORs these, never inspects
    them, so widening the word count changes no matching decision).

    Returns (offsets[M+1] int64, neighbor[E] int64, obs[E, _OBSW] uint64).
    """
    M = matcher.M
    be = matcher._boundary_edges
    ne = matcher._normal_edges
    offsets = np.zeros(M + 1, dtype=np.int64)
    nbr, qubits = [], []
    for r in range(M):
        for j in be[r]:
            nbr.append(-1)
            qubits.append(j)
        for other, j in sorted(ne[r], key=lambda x: x[1]):
            nbr.append(other)
            qubits.append(j)
        offsets[r + 1] = len(nbr)
    # Word count wide enough for the largest qubit id, but never below _OBSW so the
    # kernel path (N <= 64*_OBSW) keeps its exact [E, _OBSW] shape. When N > 64*_OBSW the
    # caller discards this obs (CPU fallback), so widening here just avoids an overflow.
    words = max(_OBSW, (max(qubits) // 64) + 1) if qubits else _OBSW
    obs = np.zeros((len(qubits), words), dtype=np.uint64)
    for e, j in enumerate(qubits):
        obs[e, j // 64] = np.uint64(1) << np.uint64(j % 64)
    return (
        offsets.astype(np.int64),
        np.asarray(nbr, dtype=np.int64),
        obs,
    )


class create(_MwpmPy):
    """MWPM decoder running the bit-exact CUDA blossom kernel (one thread per shot).

    The N > 64 path reconstruction uses the ``num_workers`` process pool, which
    starts with forkserver once CUDA is initialised, so a user script on this
    path needs an ``if __name__ == '__main__':`` guard.
    """

    def __init__(self, decoding_cfg, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)  # device/dtype/bundle + graph + matcher
        if not torch.cuda.is_available():
            raise RuntimeError("mwpm_gpu_cuda requires a CUDA GPU.")
        self._kernel = None if self.weights != "uniform" else _load_kernel()

        self.M = self.matcher.M
        self.N = self.matcher.N
        self._use_kernel = self.N <= 64 * _OBSW
        self._V_c_col_np = self.V_c_col.detach().cpu().numpy()  # [M, D], padded with dummy column N
        if not self._use_kernel:
            logger.warning(
                "CUDA path is known buggy for large H: above 256 columns the kernel is unused and "
                "the whole batch runs the serial single-process CPU matcher with the pool unused; "
                "uniform weights compile the kernel before the N check (~28s wasted); logs "
                "'CUDA blossom kernel' ready on this path."
            )
            logger.warning(
                f"mwpm_gpu_cuda: N={self.N} > 64*OBSW={64 * _OBSW}; falling back to the CPU "
                "blossom. (Raise _OBSW here and OBSW in mwpm_gpu_kernel.cu together.)"
            )
        off, nbr, obs = _build_csr(self.matcher)
        dev = self.device
        self._g_off = torch.as_tensor(off, dtype=torch.int64, device=dev)
        self._g_nbr = torch.as_tensor(nbr, dtype=torch.int64, device=dev)
        # obs is [E, _OBSW] uint64 bit-patterns (word j//64 holds bit j%64 for qubit j).
        # Reinterpret to int64 (no torch.uint64 dependency); the kernel casts each row back
        # to an OBSW-word uint64 `Obs`. Row-major [E, _OBSW] == the kernel's Obs* stride.
        self._g_obs = (
            torch.as_tensor(obs.view(np.int64), dtype=torch.int64, device=dev)
            if self._use_kernel
            else None
        )
        if self.weights != "uniform" and self._use_kernel:
            self._prepare_weighted_csr()
        self.algo = "mwpm_gpu"
        logger.info(f"mwpm_gpu_cuda decoder ready (CUDA blossom kernel, {self.device}).")

    def _prepare_weighted_csr(self):
        """Group original columns by endpoint pair; per-shot winners are picked on GPU."""
        _, _, boundary, normal = _parse_H(*self._H_coo)
        groups, nodes, neighbors = [], [], []
        for node in range(self.M):
            by_neighbor = {}
            if boundary[node]:
                by_neighbor[-1] = boundary[node]
            for other, column in normal[node]:
                by_neighbor.setdefault(other, []).append(column)
            for other, columns in by_neighbor.items():
                groups.append(sorted(columns))
                nodes.append(node)
                neighbors.append(other)
        candidates = np.full((len(groups), max(map(len, groups), default=1)), self.N)
        for edge, columns in enumerate(groups):
            candidates[edge, :len(columns)] = columns
        self._edge_candidates = torch.as_tensor(candidates, device=self.device)
        self._edge_nodes = torch.tensor(nodes, device=self.device, dtype=torch.int64)
        self._edge_neighbors = torch.tensor(neighbors, device=self.device, dtype=torch.int64)

    def _weighted_csr(self, weights):
        """Collapse raw weights in insertion order, then discretize like PyMatching 2.4."""
        padded = torch.cat((weights, weights.new_full((len(weights), 1), float("inf"))), dim=1)
        pick = padded[:, self._edge_candidates].argmin(dim=2)
        columns = self._edge_candidates.expand(len(weights), -1, -1).gather(2, pick.unsqueeze(2)).squeeze(2)
        neighbors = self._edge_neighbors.expand(len(weights), -1).contiguous()
        retained = weights.gather(1, columns)
        if retained.shape[1]:
            integral = (retained == retained.floor()).all(dim=1, keepdim=True)
            maximum = retained.amax(dim=1, keepdim=True)
            denominator = torch.where(integral, 1.0, maximum)
            normalization = torch.where(integral, 1.0, ((1 << 24) - 1) / denominator)
            scaled = retained * normalization
            lower = scaled.floor()
            edge_weights = 2 * (lower + (scaled - lower >= 0.5)).to(torch.int64)
        else:
            edge_weights = retained.to(torch.int64)
        obs = torch.zeros((*columns.shape, _OBSW), dtype=torch.int64, device=self.device)
        obs.scatter_(2, (columns // 64).unsqueeze(2), (torch.ones_like(columns) << (columns % 64)).unsqueeze(2))
        return neighbors, obs, edge_weights

    def _cpu_decode(self, synd_np_row, weights=None):
        """Exact CPU fallback using the failed shot's column weights."""
        matcher = self.matcher if weights is None else NativeMatcher.from_check_matrix(*self._H_coo, weights=weights)
        return matcher.decode(synd_np_row)

    def _decode(self, io_dict):
        dev, dt = self.device, self.dtype
        synd = io_dict["synd"]
        B, M = synd.shape
        self.batch_size = B
        N = self.N

        preflip = column_weights = weights_np = None
        if self.weights != "uniform":
            key = "llr0" if self.weights == "prior" else "llr"
            llr = io_dict.get(key)
            if not isinstance(llr, torch.Tensor) or llr.shape != (B, N) or llr.is_complex():
                raise ValueError(f"weights={self.weights!r} requires a real {key} tensor with shape [B, N].")
            llr = llr.detach().to(device=dev, dtype=torch.float64)
            if not bool(torch.isfinite(llr).all()):
                raise ValueError(f"weights={self.weights!r} requires finite {key} values.")
            column_weights = llr.abs()
            if bool((column_weights > (1 << 24) - 1).any()):
                raise ValueError("Edge weights exceed the PyMatching backend limit of 16777215.")
            preflip = (llr <= 0).to(torch.uint8)
            padded = torch.cat((preflip, preflip.new_zeros((B, 1))), dim=1)
            toggles = padded[:, self.V_c_col].sum(dim=2).bitwise_and(1).to(torch.uint8)
            synd_u8 = (synd.to(device=dev) != 0).to(torch.uint8) ^ toggles
            synd_np = synd_u8.cpu().numpy()
        else:
            synd_np = (synd.detach().cpu().numpy() != 0).astype(np.uint8)
            synd_u8 = torch.as_tensor(synd_np, dtype=torch.uint8, device=dev).contiguous()
        e_v = np.zeros((B, N), dtype=np.uint8)

        if self._use_kernel:
            if self._kernel is None:
                self._kernel = _load_kernel()
            args = (self._g_off, self._g_nbr, self._g_obs, int(M), int(N), synd_u8)
            if column_weights is not None:
                neighbors, obs, edge_weights = self._weighted_csr(column_weights)
                args = (self._g_off, neighbors, obs, int(M), int(N), synd_u8, edge_weights)
            out_mask, out_err, out_mef, out_met, out_menum = self._kernel.mwpm_decode(*args)
            err = out_err.detach().cpu().numpy().astype(np.int64)  # [B]
            if N <= 64:
                # obs_mask correction path (mwpm_gpu.py:1642-1656). Expand the OBSW-word mask ->
                # bits: qubit q lives in word q//64 at bit q%64.
                mask = (
                    out_mask.detach().cpu().numpy().view(np.uint64)
                )  # int64 bits -> uint64 [B, _OBSW]
                q = np.arange(N)
                words = mask[:, q // 64]  # [B, N] pick the owning word per qubit
                shifts = (q % 64).astype(np.uint64)  # [N]
                e_v = ((words >> shifts[None, :]) & np.uint64(1)).astype(np.uint8)
            else:
                # N>64: reconstruct explicit shortest paths from the kernel's match edges via
                # mwpm_gpu.py's SearchFlooder (mwpm_gpu.py:1657-1672). This is the ONLY path that is
                # bit-exact for N>64; the obs_mask picks a different (equal-weight) rep.
                mef = out_mef.detach().cpu().numpy().astype(np.int64)  # [B, MECAP]
                met = out_met.detach().cpu().numpy().astype(np.int64)  # [B, MECAP]
                menum = out_menum.detach().cpu().numpy().astype(np.int64)  # [B]
                e_v = np.zeros((B, N), dtype=np.uint8)
                if column_weights is not None:
                    weights_np = column_weights.cpu().numpy()
                rows = np.flatnonzero((err == 0) & (menum > 0))
                pool = self._get_pool(B) if rows.size else None
                if pool is not None:
                    shots = (
                        (mef[b, :menum[b]], met[b, :menum[b]],
                         None if weights_np is None else weights_np[b])
                        for b in rows
                    )
                    e_v[rows] = list(pool.map(
                        _mp_worker_reconstruct, shots,
                        chunksize=max(1, len(rows) // (self._num_workers * 4)),
                    ))
                else:
                    for b in rows:
                        matcher = self.matcher if weights_np is None else NativeMatcher.from_check_matrix(
                            *self._H_coo, weights=weights_np[b]
                        )
                        e_v[b] = _corr_from_match_edges(
                            matcher, mef[b, :menum[b]], met[b, :menum[b]]
                        )
            bad_err = np.nonzero(err != 0)[0]
            pred = np.pad(e_v, ((0, 0), (0, 1)))[:, self._V_c_col_np].sum(axis=2) & 1  # [B, M]
            bad_inv = np.nonzero((pred.astype(np.uint8) != synd_np).any(axis=1))[0]
            bad = np.union1d(bad_err, bad_inv)
            if bad.size:
                logger.warning(
                    f"mwpm_gpu_cuda: {bad.size} shot(s) fell back to the exact CPU blossom "
                    f"({bad_err.size} arena-cap, {bad_inv.size} failed the syndrome check)."
                )
                if column_weights is not None and weights_np is None:
                    weights_np = column_weights.cpu().numpy()
                for b in bad:
                    e_v[b] = self._cpu_decode(synd_np[b], None if weights_np is None else weights_np[b])
        elif self.weights == "prior":
            e_v = self._decode_prior(synd_np, column_weights.cpu().numpy())
        else:
            weights_np = None if column_weights is None else column_weights.cpu().numpy()
            for b in range(B):
                e_v[b] = self._cpu_decode(synd_np[b], None if weights_np is None else weights_np[b])

        e_v_t = torch.from_numpy(e_v).to(device=dev, dtype=dt)
        if preflip is not None:
            e_v_t = (e_v_t.to(torch.uint8) ^ preflip).to(dt)
        llr = (1.0 - 2.0 * e_v_t).to(device=dev, dtype=dt)
        converge = torch.ones(B, dtype=torch.int64, device=dev)
        iters = (
            torch.from_numpy(synd_np.sum(1).astype(np.int64))
            .clamp(min=1)
            .to(device=dev)
        )

        io_dict.update({"e_v": e_v_t, "iter": iters, "llr": llr, "converge": converge})
        return io_dict
