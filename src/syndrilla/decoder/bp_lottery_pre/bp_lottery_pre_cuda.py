import torch

from syndrilla.decoder.bp_lottery.bp_lottery_cuda import create as _BpLotteryCuda
from syndrilla.decoder.bp_lottery_pre.bp_lottery_pre import create as _Pre


class _SiblingExtension:
    def __init__(self, decoder, extension):
        self.decoder, self.extension = decoder, extension
        self.iteration = 0

    def __getattr__(self, name):
        return getattr(self.extension, name)

    def convergence_flag_update(self, mismatch, num_iters, converges, i):
        self.extension.convergence_flag_update(mismatch, num_iters, converges, i)
        decoder = self.decoder
        if decoder._trial_state is None:
            return
        rows = torch.arange(len(num_iters), device=num_iters.device)
        finished = decoder._record_matches((num_iters == i) & converges.bool(), rows)
        num_iters.masked_fill_(finished & (num_iters == -1), i)
        converges[decoder._stopped_siblings()] = 0


class create(_BpLotteryCuda):
    """Parallel syndrome-flip PRE retries with CUDA sibling stop."""

    _supports_pre_parallel = True
    _config_algorithm = _Pre._config_algorithm
    _config_keys = _Pre._config_keys
    _init_pre = _Pre._init_pre
    _pre_postprocess = _Pre._pre_postprocess
    _record_matches = _Pre._record_matches
    _stopped_siblings = _Pre._stopped_siblings
    _exit_hook = _Pre._exit_hook
    _stop_siblings = True
    _trial_state = None
    forward = _Pre.forward

    def __init__(self, decoding_cfg, **kwargs):
        super().__init__(decoding_cfg, **kwargs)
        self._init_pre(decoding_cfg)

    def _lottery_forward(self, io_dict):
        return super().forward(io_dict)

    def _bp_core(self, syndrome, llr0):
        extension, skip = self._ext, self._skip_converged
        try:
            if self._stop_siblings and self.early_termination and getattr(self, "_trial_owner", None) is not None:
                self._ext = _SiblingExtension(self, extension)
                self._skip_converged = True
            return _Pre._bp_core(self, syndrome, llr0)
        finally:
            self._ext, self._skip_converged = extension, skip
