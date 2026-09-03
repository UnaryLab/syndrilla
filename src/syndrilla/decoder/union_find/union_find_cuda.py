import os

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
    """JIT-compile union_find_kernel.cu on first call; cache thereafter."""
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    handles = _dll_dirs()
    decoder_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cuda_dir = os.path.join(decoder_dir, "cuda")
    src = os.path.join(cuda_dir, "union_find_kernel.cu")
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
        extra_include_paths=[cuda_dir],
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



class create(nn.Module):
    """
    CUDA-device Union-Find decoder. The whole decode runs in the CUDA extension and is
    bit-exact to ``union_find.py:decode_shot`` (toric bit-for-bit identical, surface valid).

    Accepted YAML keys (under ``decoder:``):
        check_type   : hx | hz          (default: hx)
        dtype        : float32 | float64 | float16 | bfloat16  (default: float64)
        device:
          device_type: cuda             (only cuda is supported)
          device_idx : int              (default: 0)
    """

    # cap the per-launch scratch tensor (bytes) so large batches chunk instead of OOM
    _SCRATCH_BUDGET_BYTES = 256 * 1024 * 1024

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        super().__init__()
        logger.info("Creating union_find_cuda decoder.")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "union_find_cuda requires a CUDA-capable GPU. Use union_find for CPU."
            )

        device_cfg = decoding_cfg.get("device", {})
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
                "union_find_cuda requires a pre-loaded MatrixBundle via `bundle`."
            )
        H_shape, _, V_c_col, H_matrix = bundle.select(self.check_type)
        self.H_shape = H_shape
        # V_c_col (parity-row -> qubit-column map) is read by the perfect syndrome
        # measurer when this decoder is decoders[0]; expose it like every decoder.
        self.V_c_col = nn.Parameter(V_c_col.to(self.device), requires_grad=False)
        self.M, self.N = int(H_shape[0]), int(H_shape[1])

        self._ext = _load_ext()
        H_int = (H_matrix.detach().cpu() != 0).to(torch.int32).contiguous()
        conn_off, conn_nbr, conn_q, vcc, V, B, M, N = self._ext.uf_build_lattice(H_int)
        assert int(M) == self.M and int(N) == self.N, "lattice shape disagrees with H"
        self.V = int(V)
        self.boundary = int(B)  # -1 for toric, else boundary vertex id (== M)

        self.conn_off = nn.Parameter(conn_off.to(self.device), requires_grad=False)
        self.conn_nbr = nn.Parameter(conn_nbr.to(self.device), requires_grad=False)
        self.conn_q = nn.Parameter(conn_q.to(self.device), requires_grad=False)
        self.vcc = nn.Parameter(vcc.to(self.device), requires_grad=False)

        self._block_size = 128
        self._scratch_ints = int(self._ext.uf_scratch_ints(self.V, self.N))
        bytes_per_shot = max(1, self._scratch_ints * 4)
        self._max_chunk = max(1, self._SCRATCH_BUDGET_BYTES // bytes_per_shot)

        self.algo = "union_find"
        self.num_max_iter = self.N + 1
        self.batch_size = 1
        logger.info(
            f"union_find_cuda decoder ready ({self.device}, V={self.V}, N={self.N}, "
            f"boundary={'yes' if self.boundary >= 0 else 'no'})."
        )

    def forward(self, io_dict: dict) -> dict:
        dev = self.device
        synd_in = io_dict["synd"]
        B, M = synd_in.shape
        self.batch_size = B
        assert M == self.M, f"syndrome width {M} != H rows {self.M}"

        synd = (synd_in != 0).to(device=dev, dtype=torch.int32).contiguous()

        # Decode in chunks so the per-launch scratch tensor stays within budget.
        parts = []
        for start in range(0, B, self._max_chunk):
            chunk = synd[start : start + self._max_chunk].contiguous()
            parts.append(
                self._ext.uf_serial_exact(
                    chunk,
                    self.conn_off,
                    self.conn_nbr,
                    self.conn_q,
                    self.vcc,
                    int(self.V),
                    int(self.N),
                    int(self.M),
                    int(self.boundary),
                    int(self._block_size),
                )
            )
        corr = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

        e_v = corr.to(device=dev, dtype=self.dtype)
        # sign-encoded soft output (+1 bit 0, -1 bit 1); UF carries no real LLR
        llr = (1.0 - 2.0 * e_v).to(device=dev, dtype=self.dtype)
        iters = synd.sum(dim=1).clamp(min=1).to(torch.int64)
        converge = torch.ones(B, dtype=torch.int64, device=dev)

        io_dict.update({"e_v": e_v, "iter": iters, "llr": llr, "converge": converge})
        return io_dict
