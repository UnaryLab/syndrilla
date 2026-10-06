from syndrilla.decoder.bp_interventions import BinaryCudaDecoder


class create(BinaryCudaDecoder):
    """Binary sum-product BP with oscillation/threshold prior switching.

    Config: max_iter, dtype, check_type, interval T (default 10), strength C (10).
    Every T completed iterations, restore base priors and set selected priors -C.
    Ties choose the smallest abs posterior then index; only the oscillating
    selection's counter resets. Without oscillations, select only the least
    reliable VN. Incoming edge messages persist across interventions.
    """

    mode = 1
    algo = "bp_ots"
