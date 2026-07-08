"""union_find_cuda.py -- CUDA port of the bit-exact Union-Find decoder.

The compute path is a real CUDA kernel (``union_find_kernel.cu``), NOT the CPU decoder:
one CUDA thread decodes one syndrome shot, faithfully reproducing ``union_find.py``'s
deterministic ``tsl::robin_map``/``robin_set`` iteration order (grow / fusion / peeling)
so the correction is **bit-for-bit identical** to the chaeyeunpark/UnionFind C++ reference
(``UnionFindPy.Decoder``), with the batch as the parallel axis.

Because Union-Find is inherently sequential per shot -- the grown clusters and the
spanning-forest peel order depend on the robin-table visitation order -- the kernel mirrors
the reference 1:1 rather than parallelising the graph within a shot. Each thread owns a
fixed-capacity int32 work arena and a robin-hood bucket arena (bump-allocated). If a shot
overflows its arena (``err != 0``) or fails the syndrome check it is redone on the batched
PyTorch decode (``union_find.create.forward``), which always emits a valid correction.

Subclasses the pure-PyTorch ``union_find.create`` for __init__ (device/dtype/bundle + the
one-time lattice build) and reuses its batched ``forward`` as the fallback, then overrides
forward() to launch the kernel.

The reference (and this port) is toric-only: every qubit column of H touches exactly two
parity rows (weight-2 columns); ``union_find.create.__init__`` already rejects weight-1
columns.

YAML algorithm key: dispatched as ``union_find`` on a ``cuda`` device -- the factory picks
this file because it is named ``union_find_cuda.py``; there is no separate algorithm key.
"""

import os

import numpy as np
import torch
from loguru import logger

from syndrilla.decoder.union_find.union_find import create as _UnionFindPy


_EXT = None  # compiled once per process


def _load_ext():
    """JIT-compile union_find_kernel.cu on first call; cache thereafter."""
    global _EXT
    if _EXT is not None:
        return _EXT
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
    _EXT = load(
        name="union_find_cuda_ext",
        sources=[kernel_src],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    logger.info("union_find_cuda kernel compiled.")
    return _EXT


def _build_csr(lattice):
    """Flatten the host-built ``Lattice`` into CSR arrays for the kernel.

    ``vertex_connections`` is already in the reference's tsl bucket order (built by
    ``build_lattice_from_parity``), so per vertex u we emit its neighbours v in that exact
    order together with each edge's qubit index ``edge_idx(min,max)``. That ordering IS the
    grow-step fuse order, so it must be preserved byte-for-byte.

    Returns (conn_off[M+1] int64, conn_nbr[E2] int32, conn_qub[E2] int32, vcc[M] int32),
    where E2 == sum of vertex degrees == 2*(#distinct edges).
    """
    M = lattice.num_vertices
    conn_off = np.zeros(M + 1, dtype=np.int64)
    nbr, qub = [], []
    for u in range(M):
        for v in lattice.vertex_connections[u]:
            nbr.append(v)
            e = (u, v) if u < v else (v, u)
            qub.append(lattice.edge_idx(e))
        conn_off[u + 1] = len(nbr)
    return (
        conn_off.astype(np.int64),
        np.asarray(nbr, dtype=np.int32),
        np.asarray(qub, dtype=np.int32),
        np.asarray(lattice.vertex_connection_count, dtype=np.int32),
    )


class create(_UnionFindPy):
    """Union-Find decoder running the bit-exact CUDA kernel (one thread per shot)."""

    def __init__(self, decoder_cfg, **kwargs) -> None:
        super().__init__(
            decoder_cfg, **kwargs
        )  # device/dtype/bundle + lattice + batched fallback forward
        if not torch.cuda.is_available():
            raise RuntimeError(
                "union_find_cuda requires a CUDA GPU. Use union_find on CPU."
            )
        # union_find.create defaults device to cuda when available; force a concrete cuda dev.
        if str(self.device) == "cpu":
            self.device = torch.device("cuda:0")
        self._ext = _load_ext()

        self.M = self.lattice.num_vertices
        self.N = self.lattice.num_edges
        off, nbr, qub, vcc = _build_csr(self.lattice)
        dev = self.device
        self._c_off = torch.as_tensor(off, dtype=torch.int64, device=dev).contiguous()
        self._c_nbr = torch.as_tensor(nbr, dtype=torch.int32, device=dev).contiguous()
        self._c_qub = torch.as_tensor(qub, dtype=torch.int32, device=dev).contiguous()
        self._vcc = torch.as_tensor(vcc, dtype=torch.int32, device=dev).contiguous()

        # H (dense [M, N]) for on-device output validation: a valid UF correction e
        # satisfies H @ e == syndrome (mod 2); any shot that fails (arena overflow that the
        # err flag also catches, or a latent bug) is redone on the exact CPU decoder.
        # The check runs on the GPU (`self._H_gpu`); the numpy copy is only kept for the
        # rare CPU fallback path. Entries are 0/1, so float32 is exact for the mod-2 dot.
        self._H_np = np.asarray(self.H_matrix.detach().cpu().numpy()).astype(np.uint8)
        self._H_gpu = torch.as_tensor(
            self._H_np, dtype=torch.float32, device=dev
        ).contiguous()

        # Per-thread arena size (int32 slots) for the robin-hood bucket bump-allocator.
        # Measured peak for the sequential decode is a small multiple of M; this bound is
        # generous. On overflow the shot sets err and is redone on the CPU decoder.
        self._arena_cap = int(kwargs.get("arena_cap", 64 * self.M + 4096))

        self.algo = "union_find"
        logger.info(f"union_find_cuda decoder ready (CUDA kernel, {self.device}).")

    def _cpu_decode(self, synd_np_row):
        """Fallback for one shot that overflowed the kernel arena or failed the syndrome
        check: re-decode it with the batched PyTorch Union-Find (``union_find.create``'s
        forward), which always emits a valid correction. Returns an [N] uint8 vector."""
        synd_t = (
            torch.as_tensor(synd_np_row.astype(np.uint8), device=self.device)
            .view(1, -1)
            .to(self.dtype)
        )
        e_v = super().forward({"synd": synd_t})["e_v"]
        return e_v.detach().cpu().numpy().reshape(-1).astype(np.uint8)

    def forward(self, io_dict):
        dev, dt = self.device, self.dtype
        synd = io_dict["synd"]
        B, M = synd.shape
        self.batch_size = B
        N = self.N

        # Keep the whole path on the GPU: the input syndrome is already a CUDA tensor, the
        # kernel returns its outputs on the GPU, and the syndrome-check verification is a
        # tiny GPU matmul. The old wrapper round-tripped every array through CPU/numpy and
        # verified with an O(B*M*N) numpy matmul, which dwarfed the kernel (e.g. ~21 ms of
        # host work vs a ~2 ms kernel at d=13) -- that is now all on-device.
        synd_u8 = (synd != 0).to(torch.uint8).contiguous()  # [B, M] on GPU

        e_out, err = self._ext.uf_decode(
            self._c_off,
            self._c_nbr,
            self._c_qub,
            self._vcc,
            int(M),
            int(N),
            synd_u8,
            int(self._arena_cap),
        )

        # Flag shots to redo on the batched PyTorch fallback: arena overflow (err != 0) or
        # a correction that fails the syndrome check H @ e != s (mod 2). Both on the GPU.
        pred = (e_out.to(torch.float32) @ self._H_gpu.t()).to(torch.int32) & 1  # [B, M]
        bad_mask = (err != 0) | (pred != synd_u8.to(torch.int32)).any(dim=1)  # [B] bool
        n_bad = int(bad_mask.sum().item())  # single small sync; usually 0
        if n_bad:
            bad = torch.nonzero(bad_mask, as_tuple=False).flatten().cpu().numpy()
            n_arena = int((err != 0).sum().item())
            logger.warning(
                f"union_find_cuda: {n_bad} shot(s) fell back to the batched PyTorch decode "
                f"({n_arena} arena-cap, {n_bad - n_arena} failed the syndrome check)."
            )
            synd_bad = synd_u8[bad].cpu().numpy()  # only the bad rows leave the GPU
            fixes = np.stack([self._cpu_decode(row) for row in synd_bad])  # [n_bad, N]
            e_out[torch.as_tensor(bad, device=dev)] = torch.as_tensor(
                fixes, dtype=e_out.dtype, device=dev
            )

        e_v_t = e_out.to(dtype=dt)
        # Sign-encoded soft output (+1 where bit 0, -1 where bit 1), matching union_find.py.
        llr = 1.0 - 2.0 * e_v_t
        converge = torch.ones(B, dtype=torch.int64, device=dev)
        iters = synd_u8.sum(1).to(torch.int64).clamp(min=1)

        io_dict.update({"e_v": e_v_t, "iter": iters, "llr": llr, "converge": converge})
        return io_dict
