from syndrilla.decoder.bp_interventions import BinaryCudaDecoder


class create(BinaryCudaDecoder):
    """Plain min-sum with sign-change erasures of outgoing edge messages.

    Config: max_iter, dtype (float32/float64), check_type.
    Compare with the previously SENT message: an erased zero permits revival.
    """

    mode = 2
    algo = "bp_self_corrected_min_sum"
