# Syndrome module

The syndrome YAML file defines how syndromes are measured from the injected errors.
Syndrilla currently supports three syndrome measurement models: `perfect` and `phenomenological` for the alist-based workflow, and `stim` for the circuit-level workflow.

## 1. Perfect model
The perfect model assumes ideal (error-free) syndrome measurement: the syndrome is computed directly as `H * e mod 2` with no measurement noise.
An example syndrome configuration file using the perfect model is provided in ```perfect.syndrome.yaml```:

```
syndrome:
  measure: perfect
```

The following table details the configuration parameters used in the perfect syndrome YAML file.
| Key                 | Description                                           | Example   |
|---------------------|-------------------------------------------------------|-----------|
| `syndrome.measure`  | Model for syndrome measurement                        | `perfect` |

The output syndrome is stored in `syndrome_actual` and used directly as the decoder input.

## 2. Phenomenological noise model
For alist-based simulations (not stim), the phenomenological syndrome measurer adds measurement noise to otherwise perfect syndromes.
An example syndrome configuration file using the phenomenological model is provided in ```phenomenological.syndrome.yaml```:

```
syndrome:
  measure: phenomenological
  rounds: 10
  measurement_error_rate: 0.05
```

The following table details the configuration parameters used in the phenomenological syndrome YAML file.
| Key                              | Description                                                                                         | Example            |
|----------------------------------|-----------------------------------------------------------------------------------------------------|--------------------|
| `syndrome.measure`               | Model for syndrome measurement                                                                      | `phenomenological` |
| `syndrome.rounds`              | Number of syndrome measurement rounds; each is measured from that round's own error, defaults to `1` | `10`               |
| `syndrome.measurement_error_rate`| Per-bit bit-flip probability applied independently to each round, defaults to `0.0`                 | `0.05`             |

When `rounds > 1`, the BSC error model emits a per-round error tensor `[B, rounds, N]` whose data flips are **cumulative** across rounds (see [error.md](error.md) §1). `rounds` is read from this syndrome config and propagated to the error model as `error_model.rounds`, but only `bsc` in 1-channel mode acts on it: `bsc_train` rejects `rounds > 1`, and `depol` and `stim_circuit` have no round handling, so the value is set and ignored. Pairing `rounds > 1` with any of those leaves the error model 2-D while the measurer expects a rounds axis.

The measurer then:
- For a per-round `[B, rounds, N]` error: computes the noiseless per-round syndrome `H · e_t (mod 2)` for each round directly (no replication — every round already carries its own accumulated data error), then independently flips each measured bit with probability `measurement_error_rate`, producing `[B, rounds, M]`.
- For a 2-D `[B, N]` error (`rounds == 1`): computes one syndrome `H · e (mod 2)` and applies the same measurement noise, producing `[B, M]`.

The noiseless per-round syndrome is stored in `syndrome_actual` for analysis (unaffected by `measurement_error_rate`).

### 2.1. Prior adjustment (`adjust_llr0`)
Because a flipped syndrome bit is statistically indistinguishable from an extra data flip, the measurer also exposes `adjust_llr0(llr0)`, which the pipeline calls to fold the measurement-error rate `q` into the per-data-qubit channel prior before decoding. It inflates the data-error probability `p` (recovered from `llr0 = log((1 - p)/p)`) to the effective `p_eff = p + q − 2·p·q` and rebuilds the prior. This keeps the decoder's prior physically consistent with the noisy syndromes; a zero rate leaves `llr0` untouched.

## 3. Stim model
The stim syndrome measurer samples syndromes (and observable flips) directly from the compiled stim circuit; it is the matrix loader and the stim error model that read the circuit's detector error model. Used together with the stim interface, error model, and matrix loader (see [interface.md](interface.md)).
An example syndrome configuration file using the stim model is provided in ```stim_generated.syndrome.yaml```:

```
syndrome:
  measure: stim
  rounds: 3
  measurement_error_rate: 0.1
```

The following table details the configuration parameters used in the stim syndrome YAML file.
| Key                                | Description                                                                                                            | Example  |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------|----------|
| `syndrome.measure`                 | Model for syndrome measurement                                                                                          | `stim`   |
| `syndrome.rounds`                | Number of QEC rounds baked into the generated stim circuit                                                             | `3`      |
| `syndrome.measurement_error_rate`  | Per-measurement bit-flip probability. Forwarded to `before_measure_flip_probability` when the stim circuit is generated | `0.1`    |

Unlike the phenomenological measurer, this one produces **no rounds axis**. The generated circuit's detectors already span every round, so one shot is taken per batch element and the syndrome is always `[B, num_detectors]`, with `observable_flips` at `[B, num_observables]`, whatever `rounds` is set to. The measurer therefore stores the value as `qec_rounds` rather than `rounds`, which keeps the rest of the pipeline, including `error_model.rounds`, on a round count of 1.

The stim interface assembles the circuit from `interface.yaml` (`code`, `distance`), `syndrome.yaml` (`rounds` → stim `rounds`, `measurement_error_rate` → `before_measure_flip_probability`), and `error.yaml` (the four `after_*`/`before_*` noise rates; see [error.md](error.md) §4). When both `syndrome.measurement_error_rate` and `error.before_measure_flip_probability` are set, the syndrome value wins and a warning is logged. The decoder's DEM-derived LLR priors automatically reflect whichever rate ends up in the circuit, so the decoder remains physically consistent with the sampled syndromes.