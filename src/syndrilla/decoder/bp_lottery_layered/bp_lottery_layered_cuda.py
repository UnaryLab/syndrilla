from syndrilla.decoder.bp_lottery_layered.bp_lottery_layered import create as _Lottery
from syndrilla.decoder.bp_norm_min_sum_layered.bp_norm_min_sum_layered_cuda import create as _LayeredCuda


class create(_Lottery, _LayeredCuda):
    """Transient lottery flips with the layered CUDA sweep and convergence kernels."""
