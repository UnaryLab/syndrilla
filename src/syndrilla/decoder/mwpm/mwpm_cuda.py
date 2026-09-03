import os

import numpy as np
import torch
from loguru import logger

from syndrilla.decoder.mwpm.mwpm import create as _MwpmPy

_KERNEL = None  # compiled once per process

# Observable-mask width, in 64-bit words. MUST equal OBSW in cuda/mwpm_kernel.cu. The kernel
# packs each shot's correction into OBSW words, so the GPU obs_mask path supports N (qubits ==
# H columns) up to 64*_OBSW. 4 -> N<=256 (surface d<=11); above that the host falls back to CPU.
_OBSW = 4


def _load_kernel():
    """JIT-compile mwpm_kernel.cu on first call; cache thereafter."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from torch.utils.cpp_extension import load

    decoder_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_src = os.path.join(decoder_dir, "cuda", "mwpm_kernel.cu")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    logger.info(
        "Compiling mwpm_cuda kernel -- first use in each Python environment runs nvcc "
        "(a few minutes, no progress output); later runs load from cache."
    )
    _KERNEL = load(
        name="mwpm_cuda_kernel",
        sources=[kernel_src],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    logger.info("mwpm_cuda kernel compiled.")
    return _KERNEL


def _build_csr(matcher):
    """CSR of the detector graph in ``mwpm.py``'s exact neighbor order (_build_nodes,
    mwpm.py:1470-1487): per node, boundary edges first (neighbor = -1), then normal
    edges sorted by qubit/column id. This ordering *is* the tie-break, so it must match
    mwpm.py byte-for-byte.

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
    """MWPM decoder running the bit-exact CUDA blossom kernel (one thread per shot)."""

    def __init__(self, decoding_cfg, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)  # device/dtype/bundle + graph + matcher
        if not torch.cuda.is_available():
            raise RuntimeError("mwpm_cuda requires a CUDA GPU.")
        self._kernel = _load_kernel()

        self.M = self.matcher.M
        self.N = self.matcher.N
        self._use_kernel = self.N <= 64 * _OBSW
        self._H_np = np.asarray(self.H_matrix.detach().cpu().numpy()).astype(np.uint8)
        if not self._use_kernel:
            logger.warning(
                f"mwpm_cuda: N={self.N} > 64*OBSW={64 * _OBSW}; falling back to the CPU "
                "blossom. (Raise _OBSW here and OBSW in mwpm_kernel.cu together.)"
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
        self.algo = "mwpm"
        logger.info(f"mwpm_cuda decoder ready (CUDA blossom kernel, {self.device}).")

    def _cpu_decode(self, synd_np_row):
        """Exact CPU fallback for one shot (mwpm.py's NativeMatcher)."""
        return self.matcher.decode(synd_np_row)  # uint8 [N]

    def _corr_from_match_edges(self, mef_row, met_row, n):
        """Reconstruct a correction from the kernel's match edges (N>64 path).

        PyMatching / mwpm.py extract the correction differently once num_observables > 64:
        instead of the flood-accumulated observable mask (which picks a DIFFERENT, though
        equal-weight, degenerate representative), it reconstructs the explicit shortest qubit
        path between each matched detector pair with the SearchFlooder (mwpm.py:1657-1672).
        The kernel already produces the SAME matching (identical to mwpm.py -- verified
        bit-exact via the obs_mask on N<=64), so we only need to emit its match edges and run
        mwpm.py's own, already-bit-exact SearchFlooder here. That makes the N>64 correction
        byte-for-byte identical to mwpm.py (e-diff == 0), which the obs_mask path cannot be.

        mef_row[k], met_row[k] are detector node ids (met == -1 means the boundary, which
        path_obs reads as SIZE_MAX/None).
        """
        sf = self.matcher._search_flooder
        corr = np.zeros(self.N, dtype=np.uint8)
        for k in range(int(n)):
            faults = sf.path_obs(int(mef_row[k]), int(met_row[k]))
            for f in faults:
                corr[f] ^= 1
        return corr

    def forward(self, io_dict):
        dev, dt = self.device, self.dtype
        synd = io_dict["synd"]
        B, M = synd.shape
        self.batch_size = B
        N = self.N

        synd_np = (synd.detach().cpu().numpy() != 0).astype(np.uint8)  # [B, M]
        e_v = np.zeros((B, N), dtype=np.uint8)

        if self._use_kernel:
            synd_u8 = torch.as_tensor(
                synd_np, dtype=torch.uint8, device=dev
            ).contiguous()
            out_mask, out_err, out_mef, out_met, out_menum = self._kernel.mwpm_decode(
                self._g_off, self._g_nbr, self._g_obs, int(M), int(N), synd_u8
            )
            err = out_err.detach().cpu().numpy().astype(np.int64)  # [B]
            if N <= 64:
                # obs_mask correction path (mwpm.py:1642-1656). Expand the OBSW-word mask ->
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
                # mwpm.py's SearchFlooder (mwpm.py:1657-1672). This is the ONLY path that is
                # bit-exact for N>64; the obs_mask picks a different (equal-weight) rep.
                mef = out_mef.detach().cpu().numpy().astype(np.int64)  # [B, MECAP]
                met = out_met.detach().cpu().numpy().astype(np.int64)  # [B, MECAP]
                menum = out_menum.detach().cpu().numpy().astype(np.int64)  # [B]
                e_v = np.zeros((B, N), dtype=np.uint8)
                for bi in range(B):
                    if (
                        err[bi] == 0
                    ):  # overflowed shots handled by the CPU fallback below
                        e_v[bi] = self._corr_from_match_edges(
                            mef[bi], met[bi], menum[bi]
                        )
            bad_err = np.nonzero(err != 0)[0]
            pred = (e_v.astype(np.int64) @ self._H_np.T.astype(np.int64)) & 1  # [B, M]
            bad_inv = np.nonzero((pred.astype(np.uint8) != synd_np).any(axis=1))[0]
            bad = np.union1d(bad_err, bad_inv)
            if bad.size:
                logger.warning(
                    f"mwpm_cuda: {bad.size} shot(s) fell back to the exact CPU blossom "
                    f"({bad_err.size} arena-cap, {bad_inv.size} failed the syndrome check)."
                )
                for b in bad:
                    e_v[b] = self._cpu_decode(synd_np[b])
        else:
            for b in range(B):
                e_v[b] = self._cpu_decode(synd_np[b])

        e_v_t = torch.from_numpy(e_v).to(device=dev, dtype=dt)
        llr = (1.0 - 2.0 * e_v_t).to(device=dev, dtype=dt)
        converge = torch.ones(B, dtype=torch.int64, device=dev)
        iters = (
            torch.from_numpy(synd_np.sum(1).astype(np.int64))
            .clamp(min=1)
            .to(device=dev)
        )

        io_dict.update({"e_v": e_v_t, "iter": iters, "llr": llr, "converge": converge})
        return io_dict
