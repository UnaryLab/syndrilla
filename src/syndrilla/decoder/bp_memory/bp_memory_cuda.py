from syndrilla.decoder.bp_interventions import BinaryCudaDecoder


class create(BinaryCudaDecoder):
    """Binary sum-product memory BP; fixed alpha, unscaled extrinsic subtraction.

    Config: max_iter, dtype (float32/float64), check_type, alpha (default 1).
    This adapts the MBP feedback equation to a binary syndrome graph, not BP4.
    """

    mode = 0
    algo = "bp_memory"
