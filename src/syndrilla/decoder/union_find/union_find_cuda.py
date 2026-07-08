"""union_find_cuda.py -- CUDA port of the bit-exact Union-Find decoder in union_find.py.

The compute path is a real CUDA kernel (``cuda/union_find_kernel.cu``), NOT the batched
PyTorch tensor decode: one CUDA thread decodes one syndrome shot, faithfully reproducing
the reference chaeyeunpark/UnionFind (Delfosse-Nickerson arXiv:1709.06218) tsl robin-map
iteration order, with the batch as the parallel axis for speedup. See the kernel header
for the arena/robin-table layout.

Scope: the kernel is **toric-only** (every qubit column of H has weight exactly two, so the
detector graph has no boundary vertex -- ``is_pure_toric``). That is the reference's domain
and what ``union_find_kernel.cu`` reproduces. For graphlike-but-not-toric codes (open
boundaries, e.g. the surface code -- weight-1 columns), the kernel does not apply and this
module falls back to the exact batched PyTorch decode in ``union_find.py`` for every shot.
Individual toric shots whose per-thread bucket arena overflows (``err != 0``), or whose
correction fails the syndrome check ``H @ e == s (mod 2)``, are likewise re-decoded on the
PyTorch path, so the output is always valid.

Subclasses the pure-PyTorch ``union_find.create`` for __init__ (device/dtype/bundle, the
one-time lattice build, ``V_c_col`` and ``is_pure_toric``), then overrides forward() to
launch the kernel on the toric path.

YAML algorithm key: dispatched as ``union_find`` on a ``cuda`` device -- the factory
(_create_one_decoder in decoder/decoder.py) picks this file because it is named
``union_find_cuda.py``; there is no separate algorithm key.
"""

import os

import numpy as np
import torch
from loguru import logger

from syndrilla.decoder.union_find.union_find import (
    create as _UnionFindPy,
    build_lattice_from_parity,
    _make_edge,
)


_KERNEL = None  # compiled once per process


def _load_kernel():
    """JIT-compile union_find_kernel.cu on first call; cache thereafter."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from torch.utils.cpp_extension import load

    decoder_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_src = os.path.join(decoder_dir, "cuda", "union_find_kernel.cu")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    logger.info(
        "Compiling union_find_cuda kernel -- first use in each Python environment runs "
        "nvcc (a few minutes, no progress output); later runs load from cache."
    )
    _KERNEL = load(
        name="union_find_cuda_kernel",
        sources=[kernel_src],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    logger.info("union_find_cuda kernel compiled.")
    return _KERNEL


# module-level alias so tests importing ``_load_ext`` (mwpm/osd naming) keep working
_load_ext = _load_kernel


def _build_csr(H_np):
    """CSR of the toric detector graph in union_find.py's exact tsl-bucket connection order.

    ``build_lattice_from_parity`` (union_find.py) reproduces the reference's
    ``tsl::robin_map`` edge iteration order, and its ``vertex_connections[u]`` lists each
    vertex's neighbours in exactly that order -- the order the grow/fusion step visits them,
    and therefore the tie-break the kernel must match byte-for-byte. Per vertex we emit each
    neighbour ``v`` and the qubit index of the connecting edge ``(min(u,v), max(u,v))``.

    Returns (conn_off[M+1] int64, conn_nbr[E2] int32, conn_qub[E2] int32, vcc[M] int32),
    where M is the number of detector vertices and E2 == sum of vertex degrees.
    """
    lattice = build_lattice_from_parity(H_np)
    M = lattice.num_vertices
    conns = lattice.vertex_connections
    e2q = lattice.edge_to_qubit

    conn_off = np.zeros(M + 1, dtype=np.int64)
    nbr, qub = [], []
    for u in range(M):
        for v in conns[u]:
            nbr.append(v)
            qub.append(e2q[_make_edge(u, v)])
        conn_off[u + 1] = len(nbr)

    return (
        conn_off,
        np.asarray(nbr, dtype=np.int32),
        np.asarray(qub, dtype=np.int32),
        np.asarray(lattice.vertex_connection_count, dtype=np.int32),
    )


class create(_UnionFindPy):
    """Union-Find decoder running the bit-exact CUDA kernel (one thread per shot)."""

    def __init__(self, decoder_cfg, **kwargs) -> None:
        super().__init__(
            decoder_cfg, **kwargs
        )  # device/dtype/bundle + lattice + V_c_col
        if not torch.cuda.is_available():
            raise RuntimeError("union_find_cuda requires a CUDA GPU.")
        self._kernel = _load_kernel()

        # The kernel only reproduces the toric reference (no boundary vertex). Non-toric
        # graphlike codes (surface, open boundaries) take the exact PyTorch fallback.
        self._use_kernel = bool(self.is_pure_toric)

        if self._use_kernel:
            H_np = (self.H_matrix.detach().cpu().numpy() != 0).astype(np.uint8)
            self._H_np = H_np  # [M, N] for the on-device syndrome-check safety net
            off, nbr, qub, vcc = _build_csr(H_np)
            dev = self.device
            self._g_off = torch.as_tensor(
                off, dtype=torch.int64, device=dev
            ).contiguous()
            self._g_nbr = torch.as_tensor(
                nbr, dtype=torch.int32, device=dev
            ).contiguous()
            self._g_qub = torch.as_tensor(
                qub, dtype=torch.int32, device=dev
            ).contiguous()
            self._g_vcc = torch.as_tensor(
                vcc, dtype=torch.int32, device=dev
            ).contiguous()

            # Per-thread robin-hood bucket arena (int32 slots). Generous vs the O(M) live
            # bucket regions a shot allocates; overflow trips err and that shot falls back.
            self._arena_cap = max(4096, 128 * self.M)
            # Bound per-launch device memory (~256 MB of int32) by chunking the batch.
            per_shot = self._arena_cap + int(
                15 * self.M + 4 * self.N + 9
            )  # arena + work
            self._chunk = max(1, 64_000_000 // per_shot)
        else:
            logger.warning(
                "union_find_cuda: H is graphlike but not pure-toric (open boundaries) -- "
                "the toric kernel does not apply; using the exact PyTorch Union-Find decode "
                "for every shot."
            )

        self.algo = "union_find"
        logger.info(
            f"union_find_cuda decoder ready ({self.device}, "
            f"{'CUDA kernel' if self._use_kernel else 'PyTorch fallback'})."
        )

    def _pytorch_decode(self, synd_rows):
        """Exact batched PyTorch fallback for a sub-batch; returns e_v uint8 [k, N]."""
        out = super().forward({"synd": synd_rows})
        return (out["e_v"].detach().cpu().numpy() != 0).astype(np.uint8)

    def forward(self, io_dict):
        if not self._use_kernel:
            return super().forward(io_dict)  # surface / non-toric: exact PyTorch decode

        dev, dt = self.device, self.dtype
        synd = io_dict["synd"]
        B, M = synd.shape
        self.batch_size = B
        N = self.N
        assert M == self.M, f"syndrome width {M} != H rows {self.M}"

        synd_np = (synd.detach().cpu().numpy() != 0).astype(np.uint8)  # [B, M]
        synd_u8 = torch.as_tensor(synd_np, dtype=torch.uint8, device=dev)

        e_v = np.zeros((B, N), dtype=np.uint8)
        err = np.zeros(B, dtype=np.int64)
        for start in range(0, B, self._chunk):
            stop = min(start + self._chunk, B)
            sub = synd_u8[start:stop].contiguous()
            e_out, err_out = self._kernel.uf_decode(
                self._g_off,
                self._g_nbr,
                self._g_qub,
                self._g_vcc,
                int(self.M),
                int(N),
                sub,
                int(self._arena_cap),
            )
            e_v[start:stop] = e_out.detach().cpu().numpy()
            err[start:stop] = err_out.detach().cpu().numpy().astype(np.int64)

        # A shot needs the exact PyTorch redo if it overflowed the arena (err != 0) or its
        # correction fails the syndrome check H @ e == s (mod 2). The kernel is bit-exact by
        # design, so this is a pure safety net that should rarely fire.
        bad_err = np.nonzero(err != 0)[0]
        pred = (e_v.astype(np.int64) @ self._H_np.T.astype(np.int64)) & 1  # [B, M]
        bad_inv = np.nonzero((pred.astype(np.uint8) != synd_np).any(axis=1))[0]
        bad = np.union1d(bad_err, bad_inv)
        if bad.size:
            logger.warning(
                f"union_find_cuda: {bad.size} shot(s) fell back to the exact PyTorch "
                f"Union-Find ({bad_err.size} arena-cap, {bad_inv.size} failed the "
                f"syndrome check)."
            )
            e_v[bad] = self._pytorch_decode(synd[torch.as_tensor(bad, device=dev)])

        e_v_t = torch.from_numpy(e_v).to(device=dev, dtype=dt)
        # sign-encoded soft output (+1 bit 0, -1 bit 1); UF carries no real LLR
        llr = (1.0 - 2.0 * e_v_t).to(device=dev, dtype=dt)
        converge = torch.ones(B, dtype=torch.int64, device=dev)
        # iter mirrors the syndrome weight per shot (clamped >= 1), as in union_find.py
        iters = (
            torch.from_numpy(synd_np.sum(1).astype(np.int64))
            .clamp(min=1)
            .to(device=dev)
        )

        io_dict.update({"e_v": e_v_t, "iter": iters, "llr": llr, "converge": converge})
        return io_dict
