# Trainer module

The trainer is a support module for the decoder: it defines what configures a training run rather than the model that run fits, namely the objective, the optimizer, and the epoch schedule.
It is built only by `syndrilla -t`, is selected with `-tr`, and is the one module a decode run never builds. The decoder it fits, and the two-yaml pair a learned architecture ships as, are documented in [decoder.md](decoder.md).

## 1. Running a training run (`-t`)

```
syndrilla -t -r=tests/test_outputs \
    -d=examples/alist/train_saq_hx.decoder.yaml \
    -m=examples/alist/surface_5.matrix.yaml \
    -e=examples/alist/bsc_train.error.yaml \
    -s=examples/alist/perfect.syndrome.yaml \
    -tr=examples/alist/train_saq_hx.training.yaml \
    -bs=256
```

The `-d` yaml is the training half of the pair, the architecture naming no weights; `-t` **ignores** a `decoder.config.checkpoint` rather than warm-starting from it, saying so in the log, because resuming a run is `-tckpt`'s job and it restores the optimizer and the schedule alongside the weights (Section 4).

The `-tr` yaml is three blocks, one per thing a run is configured by; the error rate comes from the error yaml (a `[lower, upper, points]` range draws one level per shot, so a run covers a stretch of the curve, see [error.md](error.md)) and the batch size from `-bs`.

```
training:
  loss:
    function: saq
    lambda_lc: 1.0
    lambda_lp: 0.2
    lambda_ent: 1.0
  optimizer:
    lr: 5.0e-4
    weight_decay: 5.0e-8
    min_lr: 1.0e-6
  schedule:
    epochs: 100
    test_batches: 200
    validation_batches: 20
    error_random_seed: 42
```

| Key                                   | Description                                                                    | Example            |
|---------------------------------------|--------------------------------------------------------------------------------|--------------------|
| `training.loss.function`              | Which objective supervises the run; names a module under `syndrilla/trainer/`  | `saq`  |
| `training.loss.*`                     | That objective's own settings, e.g. `saq`'s three term weights      | `lambda_lp: 0.2`   |
| `training.optimizer.*`                | What the `Trainer` builds the run's optimizer from; it fits Adam with `lr`, `weight_decay` and `min_lr` | `lr: 5.0e-4` |
| `training.schedule.epochs`            | Epochs to run                                                                   | `100`              |
| `training.schedule.test_batches`      | Training batches per epoch                                                      | `200`              |
| `training.schedule.validation_batches`| Validation batches per epoch, drawn clear of the training set                    | `20`               |
| `training.schedule.error_random_seed` | Seeds the error stream, so each epoch trains on the same batches                 | `42`               |

Each block has one reader: `loss` the trainer module, `optimizer` the decoder being trained, `schedule` the metric module.

## 2. The `saq` objective
The one objective shipped under `syndrilla/trainer/`, and what the `saq` decoder ([decoder.md](decoder.md)) is trained with. It is three terms, weighted by the three lambdas above and reported separately on every epoch and batch line:

| Term | Weight | What it supervises |
|------|--------|--------------------|
| `L_LC` | `lambda_lc` | Cross-entropy of the decoder's logical class logits against the true logical class of the error |
| `L_LP` | `lambda_lp` | The same cross-entropy on the logical *prior*, the class the embedding layer predicts before the transformer runs |
| `L_Ent` | `lambda_ent` | The per-qubit `llr`, through the GF(2) parity of the residual error over the logical operator's support |

`L_Ent` is computed in the **log domain**: its parity is the sign and minimum magnitude of the residual llrs over that support, not a product of per-bit probabilities. The probability-domain form multiplies one factor below 1 per bit in the support, which is exact on a code's handful of qubits and worthless on a circuit-level DEM's tens of fault mechanisms, where the product and its gradient both underflow and the term reports a constant `ln 2`. See [interface.md](interface.md) for the measured effect on the stim path. A decoder trained with this objective must therefore emit `logical_logits` and `logical_prior` alongside `llr`; `terms()` names what it returns in `term_names`, which is what the metric module keys the run's columns by.

## 3. Outputs
A chain trains its **last** decoder, and `-i` trains from circuit-level data the same way ([interface.md](interface.md)). The run writes into `-r`: `<stem>_best.pt`, the `state_dict` `decoder.config.checkpoint` loads; `<stem>_last.pt`, the run position `-tckpt` resumes from; and `<stem>_result.yaml`, a `train_full` summary plus the curve by column. Beside them sits the `main-<time>.log` every run writes, which carries the run's epoch and batch lines along with the rest of the toolchain's trace at `-l`; the console keeps the epoch lines as they are made. A second configuration adds files rather than overwriting the first.

## 4. Resuming (`-ckpt` and `-tckpt`)
The pair, `-ckpt <stem>_result.yaml` and `-tckpt <stem>_last.pt` with every other flag unchanged, continues a run to bit-identical weights and an identical curve; either alone is refused naming the other. State is restored from the `*_last.pt`, while the yaml is only checked: a run whose model, noise, schedule, `-bs`, loss, optimizer or selection metric moved is refused, naming every field that did. A weights-only `*_best.pt` has no fingerprint and is refused pointing at `decoder.config.checkpoint`; `-tckpt` needs `-t`.
