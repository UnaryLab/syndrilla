# CUDA BP interventions

Three independent decoders retain the supplied binary syndrome throughout one
decode. Choose `bp_memory`, `bp_ots`, or `bp_self_corrected_min_sum` as the
`decoding.algorithm`; their example configs are in `examples/alist/`. All require
a CUDA device and accept `float32` or `float64`, `check_type: hx | hz`, and
`config.max_iter` (default 50). The standard decoder factory selects the native
CUDA implementation. There is no CPU implementation or rebatch cap.

`bp_memory` uses binary sum-product check updates and a fixed `config.alpha`
(default 1). Its posterior is `prior + sum(incoming) / alpha`, and the outgoing
message subtracts the destination check's incoming message at unscaled strength.
Alpha 1 recovers ordinary binary sum-product BP. This is the binary syndrome
adaptation of [MBP's feedback equation](https://arxiv.org/pdf/2104.13659), rather
than the paper's quaternary BP4 decoder.

`bp_ots` uses binary sum-product and counts changes in each VN's hard decision,
starting from the channel-prior decision. After every `config.interval` completed
iterations (T, default 10), an unconverged sample restores its base priors and
sets the selected priors to `-config.strength` (C, default 10). It chooses the VN
with the largest positive flip count, breaking ties by smallest absolute
posterior LLR then index, and the globally least-reliable VN by absolute posterior
LLR then index. They may coincide. Only the oscillating selection's counter
resets; the other counts and all edge messages persist. If no VN has oscillated,
only the globally least-reliable VN is selected. This resolves the empty
oscillator selection in [the source pseudocode](https://par.nsf.gov/servlets/purl/10493558).
An intervention applies to the next iteration, so none runs after convergence or
after the final iteration.

`bp_self_corrected_min_sum` uses plain, unnormalized min-sum check updates. An
outgoing extrinsic message is erased if its sign opposes the message sent on that
edge in the immediately previous iteration. Previous zero permits either sign
to return; it does not remember an older nonzero sign.
[SCMS](https://arxiv.org/pdf/0803.1090) is adapted here to a binary syndrome graph.

The kernels run one CUDA block per sample and stop each sample at its first
syndrome match. Outputs are the standard `e_v`, `llr`, `iter`, `converge`, and
`defer` (always false). State is reset on each forward call. Sum-product clamps
the tanh product to `±(1 - machine epsilon)` to keep strong messages and
degree-one checks finite. Plain min-sum degree-one messages have infinite
magnitude; outgoing messages sum only the other checks to avoid `inf - inf`.
Syndrome agreement alone does not establish logical decoding success.
