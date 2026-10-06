"""CPU PyMatching backend with tensor inputs and outputs on the configured device."""

import numpy as np
import pymatching
import scipy.sparse as sp
import torch
from loguru import logger

from syndrilla.utils import parse_device_dtype

_MAX_PYMATCHING_WEIGHT = (1 << 24) - 1


class create(torch.nn.Module):
    """PyMatching v2 decoding; num_workers and mp_min_batch are accepted but unused."""

    def __init__(self, decoding_cfg, **kwargs):
        super().__init__()
        logger.info("Creating mwpm decoder (PyMatching).")

        self.weights = decoding_cfg.get("weights", "prior")
        if self.weights not in ("uniform", "posterior", "prior"):
            raise ValueError("weights must be 'uniform', 'posterior', or 'prior'.")
        self.skip_converged = decoding_cfg.get("skip_converged", self.weights == "posterior")
        if not isinstance(self.skip_converged, bool):
            raise ValueError("skip_converged must be a bool.")
        if self.skip_converged and self.weights != "posterior":
            raise ValueError("skip_converged=True requires weights='posterior'.")

        self.device, _ = parse_device_dtype(decoding_cfg)
        configured_device = (decoding_cfg.get("device") or {}).get("device_type", self.device.type)
        if configured_device != "cpu":
            logger.info("mwpm decodes on the host CPU with PyMatching and returns tensors on the configured device.")

        self.dtype = decoding_cfg.get("dtype", "float64")
        if self.dtype not in {"float32", "float64", "bfloat16", "float16"}:
            logger.warning(f"Invalid dtype <{self.dtype}>, default to float64.")
            self.dtype = "float64"
        self.dtype = torch.__dict__[self.dtype]

        self.check_type = decoding_cfg.get("check_type", "hx")
        if self.check_type.lower() not in {"hx", "hz"}:
            logger.warning(f"Invalid check type <{self.check_type}>, default to hx.")
            self.check_type = "hx"

        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "mwpm requires a pre-loaded MatrixBundle via the `bundle` kwarg."
            )
        self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = (
            (bundle.Hx_matrix if self.check_type.lower() == "hx" else bundle.Hz_matrix).get_index()
        )

        rows, cols = self.H_matrix.coalesce().indices().cpu().numpy()
        self._H_csc = sp.csc_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=self.H_shape
        )
        degree = np.diff(self._H_csc.indptr)
        invalid = np.flatnonzero(degree > 2)
        if invalid.size:
            column = int(invalid[0])
            raise ValueError(
                f"Column {column} has weight {degree[column]} > 2; H is not graphlike, so the "
                "MWPM decoder does not apply."
            )
        self.matcher = self._matching() if self.weights == "uniform" else None
        self.algo = "mwpm"
        self.batch_size = 1
        self.num_max_iter = int(np.count_nonzero(degree)) + 1
        self.cap = None
        self.cap_bypass = False
        self.cap_active_last = False
        logger.info("Complete.")

    def _matching(self, weights=None):
        if weights is not None:
            if np.any(weights > _MAX_PYMATCHING_WEIGHT):
                raise ValueError(
                    f"Edge weights exceed the PyMatching backend limit of {_MAX_PYMATCHING_WEIGHT}."
                )
        return pymatching.Matching.from_check_matrix(self._H_csc, weights=weights)

    def forward(self, io_dict):
        return self._forward_once(io_dict)

    def _forward_once(self, io_dict):
        previous = io_dict.get("converge")
        if not self.skip_converged or previous is None:
            return self._decode(io_dict)
        B = io_dict["synd"].shape[0]
        if not isinstance(previous, torch.Tensor) or previous.shape != (B,):
            raise ValueError("skip_converged requires converge with shape [B].")
        keep = previous == 1
        if not bool(keep.any()):
            return self._decode(io_dict)
        for key in ("e_v", "llr"):
            value = io_dict.get(key)
            if not isinstance(value, torch.Tensor) or value.shape != (B, self.H_shape[1]):
                raise ValueError(f"skip_converged requires {key} with shape [B, N].")
        self.batch_size = B
        if bool(keep.all()):
            io_dict.setdefault("iter", torch.ones_like(previous, dtype=torch.int64))
            return io_dict
        rows = (~keep).nonzero().flatten()
        subset = {
            key: value[rows.to(value.device)]
            if key in ("synd", "llr", "llr0", "e_v", "converge", "iter")
            and isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == B
            else value
            for key, value in io_dict.items()
        }
        decoded = self._decode(subset)
        for key in ("e_v", "llr", "converge", "iter"):
            value = io_dict.get(key, torch.ones_like(previous, dtype=torch.int64)).clone()
            value[rows.to(value.device)] = decoded[key].to(value)
            io_dict[key] = value
        self.batch_size = B
        return io_dict

    def _decode(self, io_dict):
        synd = io_dict["synd"]
        B, _ = synd.shape
        self.batch_size = B
        N = self.H_shape[1]
        synd_np = (synd.detach().cpu() != 0).numpy().astype(np.uint8)

        preflip = weights = None
        if self.weights != "uniform":
            key = "llr0" if self.weights == "prior" else "llr"
            llr = io_dict.get(key)
            if not isinstance(llr, torch.Tensor) or llr.shape != (B, N) or llr.is_complex():
                raise ValueError(f"weights={self.weights!r} requires a real {key} tensor with shape [B, N].")
            llr = llr.detach().to(device="cpu", dtype=torch.float64).numpy()
            if not np.isfinite(llr).all():
                raise ValueError(f"weights={self.weights!r} requires finite {key} values.")
            weights = np.abs(llr)
            preflip = (llr <= 0).astype(np.uint8)
            col = self.V_c_col.detach().cpu().numpy()
            toggles = np.pad(preflip, ((0, 0), (0, 1)))[:, col].sum(axis=2) & 1
            synd_np ^= toggles.astype(np.uint8)

        if weights is None:
            e_v = self.matcher.decode_batch(synd_np)
        elif self.weights == "prior" and N == 0:
            e_v = self._matching(np.empty(0, dtype=np.float64)).decode_batch(synd_np)
        elif self.weights == "prior":
            unique, inverse = torch.unique(torch.from_numpy(weights), dim=0, return_inverse=True)
            inverse = inverse.numpy()
            e_v = np.empty((B, N), dtype=np.uint8)
            for index, weight in enumerate(unique.numpy()):
                rows = np.flatnonzero(inverse == index)
                e_v[rows] = self._matching(weight).decode_batch(synd_np[rows])
        else:
            e_v = np.empty((B, N), dtype=np.uint8)
            for row, weight in enumerate(weights):
                e_v[row] = self._matching(weight).decode_batch(synd_np[row:row + 1])[0]
        if preflip is not None:
            e_v ^= preflip
        e_v = torch.from_numpy(e_v).to(device=self.device, dtype=self.dtype)
        io_dict.update(
            e_v=e_v,
            llr=1.0 - 2.0 * e_v,
            converge=torch.ones(B, dtype=torch.int64, device=self.device),
            iter=torch.from_numpy(synd_np.sum(1).astype(np.int64)).clamp(min=1).to(self.device),
        )
        return io_dict
