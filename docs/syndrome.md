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
  rounds: 3
  measurement_error_rate: 0.01
```

The following table details the configuration parameters used in the phenomenological syndrome YAML file.
| Key                              | Description                                                                                         | Example            |
|----------------------------------|-----------------------------------------------------------------------------------------------------|--------------------|
| `syndrome.measure`               | Model for syndrome measurement                                                                      | `phenomenological` |
| `syndrome.rounds`              | Number of syndrome rounds replicated from the true syndrome                                         | `3`                |
| `syndrome.measurement_error_rate`| Per-bit bit-flip probability applied independently to each replicated round                         | `0.01`             |

This computes the true syndrome (H*e), replicates it `rounds` times, and independently flips each bit with probability `measurement_error_rate`.
The true syndrome is stored in `syndrome_actual` for analysis.

## 3. Stim model
The stim syndrome measurer samples syndromes (and observable flips) directly from a stim circuit's detector error model. Used together with the stim interface, error model, and matrix loader (see [interface.md](interface.md)).
An example syndrome configuration file using the stim model is provided in ```stim_generated.syndrome.yaml```:

```
syndrome:
  measure: stim
  rounds: 1
  measurement_error_rate: 0.01
```

The following table details the configuration parameters used in the stim syndrome YAML file.
| Key                                | Description                                                                                                            | Example  |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------|----------|
| `syndrome.measure`                 | Model for syndrome measurement                                                                                          | `stim`   |
| `syndrome.rounds`                | Number of QEC rounds in the generated stim circuit and syndrome samples taken per error instance                       | `1`      |
| `syndrome.measurement_error_rate`  | Per-measurement bit-flip probability. Forwarded to `before_measure_flip_probability` when the stim circuit is generated | `0.01`   |

The stim interface assembles the circuit from `interface.yaml` (`code`, `distance`), `syndrome.yaml` (`rounds` → stim `rounds`, `measurement_error_rate` → `before_measure_flip_probability`), and `error.yaml` (the four `after_*`/`before_*` noise rates; see [error.md](error.md) §3). When both `syndrome.measurement_error_rate` and `error.before_measure_flip_probability` are set, the syndrome value wins and a warning is logged. The decoder's DEM-derived LLR priors automatically reflect whichever rate ends up in the circuit, so the decoder remains physically consistent with the sampled syndromes.