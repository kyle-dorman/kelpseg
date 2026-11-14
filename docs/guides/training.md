# Training and portable models

For the scientific rationale and evaluation scopes, see [methods](../methods.md).

Use the [tiny walkthrough](../quickstart.md) first. These sections provide detailed options and reusable recipes. Run steps sequentially and stop if a command fails. Use fresh output names.

The selected research model is the [8-band B0 BCE+Dice epoch-16 EMA baseline](../experiments.md#current-planet-8-band-b0-baseline). Generic recipe defaults below remain supported but do not identify that selected model.

## Model and training inputs (Python)

The installed package supplies manifest-selected tensors, SegFormer B0–B3
models, the binary objective, and pooled confusion helpers. The public `train`
command below connects these through Lightning. Use this notebook/Python example to
inspect the full published baseline inputs without reading test payloads:

```python
import logging
from kelpseg.paths import load_paths
from kelpseg.training.data import build_training_inputs

logging.basicConfig(level=logging.INFO)
root = load_paths("paths.local.toml").data_root
inputs = build_training_inputs(
    root / "data/chips/ca_bc/v1",
    root / "data/splits/ca_bc/v1/assignments.csv",
    selection="rgbnir",  # "all" = native eight; "rgb" = red/green/blue
    experiment="baseline",
    variant="augmented",  # primary with-BC baseline; "base" excludes BC
    seed=33,
)
print(inputs["counts"])
print(inputs["train"].band_names)  # red, green, blue, nir
sample = inputs["train"][0]
print(sample["image"].shape, sample["target"].shape, sample["metadata"])
```

For the full with-BC release, expect 18,311 candidate training windows, 8,518
eligible and 9,793 zero-kelp exclusions; validation retains 933 windows and
1,262 test identities remain metadata only. `excluded_training_rows` preserves
excluded identities. `val` uses representative disjoint windows, including
backgrounds. Neither input construction nor train/validation loading opens test
payloads; `test_rows` is deliberately a list of metadata, not a dataset.
Missing labels or training kelp counts raise errors naming the affected payload.

Each sample returns float32 `image` (channels/height/width), int64 `target`
(height/width), boolean `valid_mask`, and source/window `metadata`. Native-band
validity is detected before selecting channels. Training applies the approved
D4 + light Affine + SomeOf(2) factory to selected channels inside one
Albumentations `Compose`. Affine has a separate 50% chance, 0.95–1.05× zoom
and independent ±16-pixel shifts while keeping the 512×512 canvas.
The pipeline includes unit-scale conversion, restoration to raw units with the
shared 5000 cap, stock `Normalize` using the shared band-named constants, and
`ToTensorV2`. It restores zero invalid fill after normalization. Channel dropout
therefore always removes a model input. Augmented zero values never redefine nodata. One seeded pipeline advances
across calls; these bounded checks use ordinary single-process loading.

Targets use 0/1/-100. Nodata, padding, and coarse-dropout holes are ignored;
valid imagery with ignored labels stays image-valid but unscored. Metadata keeps
original source placement and true extents. D4 can move padding in training;
use the transformed validity mask there. Validation applies no augmentation and
keeps bottom/right padding, removable using `true_height`/`true_width`.

Construct the model and configurable objective directly:

```python
from kelpseg.models import build_model
from kelpseg.training.loss import BinaryObjective
from kelpseg.training.metrics import build_metrics

# Construct one model at a time. None avoids any pretrained-weight download.
model = build_model(
    len(inputs["train"].band_names), encoder_weights=None, encoder_name="mit_b0"
)
from kelpseg.training.settings import load_settings

settings = load_settings("configs/train/planet8.toml")
objective = BinaryObjective(settings.losses)
metrics = build_metrics()  # Native TorchMetrics MetricCollection.
# Given B1HW logits and BHW targets from later validation batches:
# loss, components = objective(logits, targets)
# metrics.update((logits[:, 0] >= 0).long(), targets)  # Each batch.
# report = metrics.compute()  # Once after the entire validation scope.
# metrics.reset()  # Before the next scope.
```

`build_model` supports SegFormer B0–B3 and returns one full-resolution logit.
The selected research model is B0; the generic default is B3.
Its default `encoder_weights="imagenet"` is the initialization for new baselines
and may download weights on first use. SMP retains RGB weights for three inputs;
for four/eight it repeats the RGB input kernels cyclically and multiplies by
3/channel-count. This positional initialization does not assign pretrained
spectral meaning to the native eight-band order.

The default objective is 0.7 BCE with 0.1 label smoothing plus 0.3 batch-pooled Lovasz
hinge on hard labels. Both ignore -100; all-ignored support returns a
differentiable zero. TorchMetrics accumulates the confusion matrix, foreground
IoU, precision, recall and F1 over updates. Explicit integer predictions
use probability >= 0.5 and avoid automatic logits/probability guessing. Each
metric ignores -100. Results are tensors; the confusion matrix is
`[[tn, fp], [fn, tp]]`. Undefined ratios use TorchMetrics' zero convention;
inspect `report["confusion_matrix"].sum()` for scored support, and never treat
zero-support output as evidence of model performance. These helpers provide no
epoch loop, optimizer policy or checkpoint selection.

## OpenEarthMap B0 initialization comparison

Use [the OpenEarthMap recipe](../../configs/train/initialization/openearthmap_b0.toml)
to test a segmentation-trained encoder **and decoder** against the
[selected ImageNet-encoder baseline](../../configs/train/model_size/b0.toml).
The recipes differ only in run name and initialization. Both use native eight
bands, with-BC assignments, the same normalization/augmentation, BCE + Dice,
seed, learning rate, batch size, 70-epoch budget, early stopping and EMA policy.
The current selected model is unchanged; no improvement is established by adding
this option. Compare validation-selected models on matching coverage, and keep
test results out of recipe/checkpoint selection.

The new recipe explicitly sets:

```toml
encoder_name = "mit_b0"
encoder_weights = "none"
model_weights = "openearthmap_b0"
```

`model_weights` is optional. The OpenEarthMap choice is `"openearthmap_b0"`;
[the ADE20K choices](#ade20k-b0-initialization-comparison) are described below.
It requires B0 and disabled ImageNet loading; incompatible choices fail before
model construction. For Python, use `encoder_weights=None` with the same
`model_weights` value. Other recipes retain their existing initialization.

Check the machine paths in `paths.local.toml`, then preview the comparison:

```bash
uv run --frozen kelpseg train --recipe configs/train/initialization/openearthmap_b0.toml --dry-run
```

The preview reads manifests, prints the resolved initialization source/revision,
inputs and budget, and creates no run or model download. When ready to launch a
new experiment, the same command without `--dry-run` starts fitting. Do not use
`--resume` to initialize from OpenEarthMap: this is a fresh run with fresh
optimizer, scheduler and EMA state.

First model construction downloads `model.safetensors` from
[`boehnen/satlens-segformer`](https://huggingface.co/boehnen/satlens-segformer/tree/19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6),
revision `19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6` (MIT). The ordinary Hugging Face
cache stores it outside Git; `HF_HOME` can relocate that cache. First use needs
network access unless the pinned artifact is already cached. Run and exported
bundle metadata record the source, revision and transfer policy. Existing
completed-epoch resume restores its saved weights without downloading or
reapplying initialization; exported bundles remain self-contained and offline.

The source model has RGB inputs and nine output classes. The importer retains
all encoder/decoder parameters and normalization buffers, discards its output
classifier, and preserves our fresh one-logit kelp head. Four/eight inputs repeat
RGB kernels cyclically and scale by 3/channel-count, matching the ImageNet
baseline policy; this is positional initialization, not a physical-band mapping.
Three inputs retain the source kernels. We retain Planet normalization rather
than applying the source RGB image processor.

This transfers weights into the **existing SMP architecture**, including its
normalization epsilon, decoder regularization and upsampling conventions. It is
not exact numerical reproduction of the original Hugging Face model. No
Transformers runtime or new prediction architecture is required.

### Run training and matching validation reports in tmux

The root-level [Bash runner](../../run_openearthmap_comparison.sh) trains the
OpenEarthMap recipe, reuses the selected ImageNet B0 epoch-16 EMA model, and
creates fresh validation predictions and reports for both models. It does not
retrain the baseline. Each phase stops on failure; it has no partial-work reuse.

From an existing tmux shell on the configured GPU host:

```bash
bash /home/sky/kelpseg/run_openearthmap_comparison.sh
```

The runner changes to its checkout directory itself. To check inputs without
starting training, prediction, report generation or a weight download:

```bash
bash /home/sky/kelpseg/run_openearthmap_comparison.sh --check
```

It reads `paths.local.toml`, uses the selected baseline under
`<data_root>/model-size-runs/`, and creates
`<data_root>/comparisons/openearthmap-b0-<UTC timestamp>/`. An optional absolute
output-directory argument overrides that destination; it must not exist.
`BASELINE_RUN` may point to another location of the same complete selected run.
Preflight checks the actual saved baseline settings against the new recipe,
matching chip/assignment paths, the epoch-16 EMA bundle and CUDA availability.
The existing recipe enables W&B at `kdorman90-ucla/kelpseg`.

Training and reporting output goes directly to the terminal, preserving live
progress in tmux. The runner does not pipe or capture console output.
The comparison directory contains `training-result.json` with
the exact new run directory, per-model report recipes,
`baseline-report/report.html`, `openearthmap-report/report.html`, and
`comparison.csv` / `comparison.md`. The last two show pooled validation metrics
and OpenEarthMap-minus-baseline differences after checking matching source/window
identities and scored support. Predictions remain available beside the reports.
Training checkpoints and the selected candidate bundle stay in the normal
`runs_root` run folder. Baseline artifacts are read-only.

Only validation data are predicted/reported. Both models were selected with
validation IoU, so the comparison is a validation diagnostic, not an independent
test estimate. Training and inference require CUDA; reporting runs on CPU.
All outputs are local until separately backed up.

## ADE20K B0 initialization comparison

Two recipes isolate whether a segmentation-trained encoder helps and whether
also transferring its decoder adds value:

| Recipe | `model_weights` | Loaded components | Fresh components |
|---|---|---|---|
| [Encoder only](../../configs/train/initialization/ade20k_b0_encoder.toml) | `ade20k_b0_encoder` | ADE20K-trained encoder | Decoder and binary kelp output |
| [Encoder and decoder](../../configs/train/initialization/ade20k_b0_full.toml) | `ade20k_b0_full` | ADE20K-trained encoder and decoder | Binary kelp output |

Both require `encoder_name = "mit_b0"` and `encoder_weights = "none"`. Both
use the **same segmentation-trained encoder**, not the ImageNet-only MiT weights.
All parameters remain trainable. The recipes otherwise match the selected
[ImageNet B0 baseline](../../configs/train/model_size/b0.toml): eight bands,
with-BC assignments, native normalization, augmentation, BCE + Dice, seed, LR,
batching, 70-epoch budget, early stopping and EMA. Each starts a new run with
fresh optimizer/scheduler/EMA state and a distinct W&B name.

Source: [`nvidia/segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/tree/489d5cd81a0b59fab9b7ea758d3548ebe99677da),
revision `489d5cd81a0b59fab9b7ea758d3548ebe99677da`, `model.safetensors`.
First use downloads through the ordinary external Hugging Face cache. The
150-class source classifier is discarded. Input kernels use the same cyclic
RGB expansion and 3/channel-count scaling as the baseline. Encoder-only transfer
preserves every initial decoder parameter/buffer; full transfer includes decoder
normalization buffers. Run/bundle metadata record which components were loaded.
SMP normalization epsilon, regularization and upsampling remain unchanged; this
is weight transfer rather than exact reproduction of Hugging Face predictions.
Saved bundles and completed-epoch resume do not need initialization downloads.

The separate [ADE20K runner](../../run_ade20k_comparison.sh) leaves the
OpenEarthMap runner unchanged. After the current experiment finishes, run it in
an existing tmux shell:

```bash
bash /home/sky/kelpseg/run_ade20k_comparison.sh
```

For read-only input/recipe/GPU checks without downloads or output creation:

```bash
bash /home/sky/kelpseg/run_ade20k_comparison.sh --check
```

It runs encoder-only training, then full training in separate processes, then
fresh validation predictions/reports for the selected ImageNet baseline and both
ADE20K models. Terminal output is direct: no `tee` or console capture. A failed
step stops the sequence; partial-output reuse is not supported. The script is
prepared for manual execution and does not automatically wait for another run.

Default results live under
`<data_root>/comparisons/ade20k-b0-<UTC timestamp>/`; an optional absolute output
argument overrides this and must be fresh. `BASELINE_RUN` can relocate the same
selected baseline. The two `*-training-result.json` files retain the exact run
paths under `runs_root`. Open `comparison.md` or `comparison.csv` for both
variants' differences from baseline and the full-minus-encoder difference.
Individual HTML reports are under `baseline-report/`, `ade20k-encoder-report/`
and `ade20k-full-report/`, with their predictions and report recipes alongside.
Matching source/window identities, scored support and threshold are required
before presenting deltas. No held-out test predictions are generated: these are
validation diagnostics for validation-selected models, not independent test
estimates. All outputs remain local until separately backed up.

## Configurable loss experiments

Recipes without `losses` retain 0.7 BCE with 0.1 smoothing + 0.3 Lovasz.
For a new experiment, copy a training recipe, change its run `name`, and append
`[[losses]]` tables **after all top-level settings**. Each term has:

- `type`: the implementation to construct (required).
- `name`: its independent logging identifier (defaults to type).
- `weight`: a positive multiplier (defaults to 1.0).
- `parameters`: options for that loss; omitted values use the defaults below.

The list replaces the default objective. Names must be unique, but types can
repeat with different parameters. Names use lowercase letters, digits and
underscores; metric names such as `loss`/`iou` and names ending `_step` are reserved.
Weights are literal multipliers, not automatically normalized; omit disabled terms.

```toml
[[losses]]
type = "boundary_bce"
name = "edge_bce"
weight = 0.7
parameters = { smoothing = 0.0, radius = 1, factor = 2.0, per_image = false }

[[losses]]
type = "dice"
name = "foreground_dice"
weight = 0.3
parameters = { smooth = 0.0, eps = 1e-7, log_loss = false, per_image = true }
```

This logs `train/edge_bce`, `val/edge_bce`, `train/foreground_dice`,
`val/foreground_dice`, and corresponding `train/<name>_step` values. Repeating
`type = "dice"` under two different names can compare or combine different pooling
choices. Component values are unweighted; total loss is the weighted sum.

| Type | Loss-specific parameters and defaults |
| --- | --- |
| `bce` | `smoothing = 0.0`, `pos_weight = 1.0`, `reduction = "mean"` |
| `boundary_bce` | BCE options plus `radius = 1`, `factor = 2.0`, `normalization = "weights"` |
| `distance_bce` | `alpha = 0.0`, `half_distance_m = 120.0`; fixed unsmoothed BCE, positive weight 1, mean reduction and batch pooling |
| `lovasz` | No extra options beyond `per_image` |
| `dice` | `smooth = 0.0`, `eps = 1e-7`, `log_loss = false` |
| `jaccard` | `smooth = 0.0`, `eps = 1e-7`, `log_loss = false` |
| `focal` | `gamma = 2.0`, `normalized = false`, `reduction = "mean"`; omitted `alpha` and `reduced_threshold` disable those options |
| `tversky` | Overlap options plus `alpha = 0.5`, `beta = 0.5`, `gamma = 1.0` |
| `focal_tversky` | Same as Tversky with default `gamma = 2.0` |

Except for `distance_bce`, whose pooling is fixed to batch mode, every type accepts `per_image = false` or `true`. False pools valid pixels within
one actual microbatch. True preserves image identity, computes each image's loss
separately, and takes their arithmetic mean. Entirely ignored images are excluded
from that mean; valid background-only images remain included. Gradient accumulation
does not enlarge the loss pooling group. All-ignored batches yield differentiable
zero. Per-image evaluation can cost more time because it calls the loss per image.

BCE smoothing maps targets to `(1-smoothing)*y + smoothing/2`; its default in an
explicit term is zero, while the historical default recipe specifies 0.1.
`pos_weight` multiplies the positive BCE term and must be positive. `reduction`
accepts `mean` or `sum` for BCE, boundary BCE and focal. With per-image evaluation,
it reduces pixels within each image; the final image aggregation is still a mean.
Scalar outputs are required for optimization. `sum` changes the objective scale
with valid-pixel count; the weights and learning rate may need reassessment.

For Dice/Jaccard/Tversky, `smooth` adds to numerator/denominator, `eps` bounds
the denominator, and `log_loss` selects negative log overlap instead of one minus
overlap. Tversky alpha weights FP and beta weights FN (each in [0,1], not both
zero). Balanced 0.5/0.5 with gamma 1 equals Dice. Tversky gamma is the literal
exponent on its loss, not its reciprocal; positive fractional exponents are allowed
with zero gradients defined at exactly zero loss. `focal_tversky` is the same
implementation with a different default exponent.

Focal alpha, if supplied, weights positives by alpha and negatives by 1-alpha.
Gamma controls focusing; zero removes it. `normalized = true` uses SMP's focal
normalization followed by the selected reduction. `reduced_threshold` must be
in (0,1) and requires `reduction = "sum"`, following the installed SMP contract.

Boundary BCE boosts pixels whose square radius neighborhood contains both supplied
classes. Neighborhoods touching ignored support or the chip edge receive no boost.
Radius is a positive integer; factor is at least 1. For mean reduction,
`normalization = "weights"` divides weighted BCE by the valid weight sum;
`"pixels"` divides by the valid pixel count. Sum reduction simply sums weighted
BCE, so normalization has no effect. Factor 1 gives ordinary BCE with matching
smoothing/positive weighting/reduction. No distance maps are computed;
distance-based boundary loss remains deferred.

Distance BCE is an optional controlled-experiment loss, not the selected baseline.
It requires chips augmented with aligned distance/land arrays using
`kelpseg prepare-chip-distances --chips /path/to/chips --shoreline
/path/to/prepared-shoreline --output /path/to/fresh-distance-chips --workers 1`.
Point the copied training recipe at that derived collection. Positive-pixel
boost is `alpha * 2**(-distance/half_distance_m)`; background and unknown distance
keep weight 1, mapped land gets full boost, and mean BCE divides by the valid
weight sum. Alpha zero gives ordinary BCE. The selected baseline remains plain
BCE plus Dice; completed distance-weighting results and historical launch commands
are in [Task 041](../../tasks/041_distance_weighted_bce_experiment.md) and the
[historical recipe reference](../reference/historical_recipes.md#run-the-b0-distance-weighted-bce-screen).

The small construction function reuses PyTorch BCE and installed
[SMP loss implementations](https://smp.readthedocs.io/en/v0.3.3/losses.html).
Shared project rules remain binary logits, -100 ignore labels and float32 loss
arithmetic. Ignore filtering happens within the selected image/batch group, which
also supplies ignore support for installed SMP Jaccard. SMP overlap losses return
zero for groups with no labeled foreground. Pair them with BCE, boundary BCE,
focal or Lovasz to penalize false positives on such groups. Per-image averaging
includes these valid zero-valued background images.

Run/settings JSON, W&B config and resume identity store resolved types, names,
weights and parameters. Changing any loss settings requires a fresh run. Resume
requires the complete current settings schema; missing historical fields are not inferred.
Epoch logs remain valid-pixel-weighted means of scalar microbatch losses, including
when a term uses per-image pooling or sum reduction. Compare experiments using
unchanged validation IoU/precision/recall and post-training diagnostics, rather
than comparing numeric losses across different objectives.

Historical loss-screen recipes and completed launch sequences are in the
[historical recipe reference](../reference/historical_recipes.md#loss-screening-recipes).

## Train and validate

Use [GPU setup](../gpu_setup.md) and restore the full chip/split release before
production training. To run the selected B0 recipe as a **new** experiment, copy
[its settings](../../configs/train/model_size/b0.toml) beside the original:

```bash
(
  set -e
  test ! -e configs/train/model_size/b0.local.toml
  cp configs/train/model_size/b0.toml configs/train/model_size/b0.local.toml
)
```

Edit `name` to a distinct descriptive value in the copy. Its existing
`paths_file = "../../../paths.local.toml"` reaches the repository's machine
settings; if you move the copy elsewhere, change that reference. Check that
`data_root` points to the full release, not the tiny walkthrough root. The B0
recipe enables W&B, so authenticate as described in GPU setup or set
`wandb = false` explicitly. Review batch size and worker settings for your host.

Preview before fitting; it reads manifests and does not create an output folder
or open image arrays. Run the fit only if the preview succeeds:

```bash
(
  set -e
  uv run --frozen kelpseg train --recipe configs/train/model_size/b0.local.toml --dry-run
  uv run --frozen kelpseg train --recipe configs/train/model_size/b0.local.toml
)
```

This starts a fresh run; it does not restore the selected epoch-16 bundle or
guarantee the same selected epoch. To use that existing model, follow
[its restore location](../experiments.md#current-planet-8-band-b0-baseline)
and the [prediction guide](prediction_reporting.md).

The four explicit recipes reuse the same native collection and assignments:

| Recipe | Ordered inputs |
|---|---|
| `planet8.toml` | coastal_blue, blue, green_i, green, yellow, red, rededge, nir |
| `planet4.toml` | red, green, blue, nir |
| `planet3.toml` | nir, red, green |
| `planet_RGB.toml` | red, green, blue |

The existing Python `selection="rgb"` still means red/green/blue. Set
`variant = "base"` in a copied recipe for the without-BC comparison; keep epochs
and optimizer policy matched. Production recipes default to `variant = "augmented"`.

`paths_file = "../../paths.local.toml"` selects your machine settings, relative
to the recipe's directory. Change it if you move the recipe, or use an absolute
path. Machine roots remain in that separate TOML so they need not be repeated
in every training recipe. Relative `chips` and `assignments` resolve against
its `data_root`. Each recipe has a `name` used as the run-ID prefix:

```toml
name = "planet3"
experiment = "baseline"
variant = "augmented"
paths_file = "../../paths.local.toml"
```

Each fit receives `<name>-<UTC timestamp>` automatically, printed with the
resolved output location. Rename a copied recipe in TOML to distinguish a
comparison; the filename itself does not determine the run name. `experiment`
selects the assignment scope and is recorded separately from the descriptive
name. The maintained training command supports `baseline`. The completed
LORO study is archived in Git; see [its history](../reference/historical_recipes.md#completed-loro-study).
Dry-run IDs are previews, not reservations;
a later fit gets its own ID. Unknown settings and empty validation produce
errors. Fresh runs refuse existing output folders; explicit resume uses the saved run identity.

For a bounded local smoke, copy `configs/train/tiny_cpu.toml`. Its `chips` and
`assignments` default to the `my-tiny-v2-01` walkthrough outputs; change both
if you chose another example name. This
recipe uses NIR/red/green, float32 CPU, no weight download, two epochs, two
training microbatches and one validation batch per epoch. It still uses the
512 canvas and approved preprocessing; limited validation is smoke evidence
only. Preview and run it:

```bash
uv run kelpseg train --recipe configs/train/tiny_cpu.toml --dry-run
uv run kelpseg train --recipe configs/train/tiny_cpu.toml
```

The generic `planet8`/`planet4`/`planet3`/`planet_RGB` recipes target CUDA bf16 mixed precision with float32 loss, 70
epochs, seed 42, microbatch 3 and accumulation 8. They use ImageNet-initialized
SegFormer B3, AdamW (3e-4, betas .9/.95, decay .01 except biases/normalization),
five warmup epochs, cosine decay to 3e-5 and norm clipping .8. Hardware sizing
and CUDA verification remain GPU-host work; do not run production on the Mac.
Workers, pinned memory and persistent workers are explicit machine settings.
The approved D4 → Affine(p=.5) → SomeOf(2), 5000 cap and exact band constants
remain fixed. Training uses positive chips; validation includes backgrounds;
test payloads are never loaded, including after fitting.

Lightning steps the scheduler once per optimizer update. Its initial LR is
zero; the scheduled LR reaches the peak after the warmup updates and the floor
after the final update. Thus the last applied update uses the preceding LR.
The final incomplete accumulation group still performs an optimizer update;
Lightning divides each microbatch loss by the configured accumulation factor,
including that final group. The preview reports actual update budget and
partial groups; effective batch 24 describes complete groups only.

Training and validation each log IoU, precision, recall, F1, AUROC and calibration
error through
native TorchMetrics collections. Lightning computes pooled epoch metrics and
resets their state; these are not averages of per-batch IoUs. `train/auroc` and
`val/auroc` use sigmoid probabilities with 257 uniform thresholds; higher is
better. `train/calibration_error` and `val/calibration_error` use 15 uniform
probability bins and L1 expected calibration error; lower is better. Both exclude
ignored pixels and log only at epoch end, using fixed-size running state.
Accuracy is no longer logged. These diagnostics do not change IoU checkpoint
selection. Training metrics
measure the augmented training pass with changing current weights, so they are
diagnostics, not directly comparable to deterministic validation with EMA.
`train/iou_step` records current-batch foreground IoU (threshold .5, ignoring
-100, zero for an empty union); it does not update or replace pooled epoch IoU.
The runner logs step metrics every optimizer update. With gradient accumulation,
the logged batch is the final microbatch of that update, not the combined group.
`train/loss_step` records each logged training batch loss; `train/loss` and
`val/loss` are native TorchMetrics valid-pixel-weighted epoch means of batch
configured losses. Overlap losses and Lovasz pool within each microbatch, not
across accumulated gradients or the whole epoch. The scalar logger receives the collection directly; confusion
matrices remain available through the Python metric helper, without hand-split
TN/FP/FN/TP log columns. Validation uses fixed threshold .5.
Aggregate foreground `val/iou` selects best weights, preserving the earliest tie.
Each configured term logs unweighted `train/<name>`, `val/<name>` and
`train/<name>_step` values. Epoch components use the same valid-pixel weighting
as combined loss, so `loss = sum(weight * component)` up to rounding. Default
recipes still use `bce` (0.1 label smoothing) and batch-pooled `lovasz` with weights
0.7 and 0.3. CSV and W&B use the same component names. See
[loss experiments](training.md#configurable-loss-experiments) for supported
terms, parameter conventions and TOML examples.
CSV and W&B share slash-separated names (`train/iou`, `val/iou`, `val/pixels`),
so W&B groups training and validation panels separately. Update counters use
`trainer/optimizer_updates`; native Lightning logging supplies the global step.
Zero scored validation support fails rather than selecting a meaningless model.
EMA (.999) first copies weights after optimizer update 101; earlier validation
uses current weights. Afterward validation and selected weights use EMA.

Each run writes resolved settings and a run record under `<runs_root>/<run_id>`,
CSV metrics under `metrics/`, and distinct best/last Lightning checkpoints under
`checkpoints/`. The checkpoint `state_dict` contains validation weights;
`current_model_state` retains ordinary weights separately from averaging state.
After fitting, `model/` contains the validation-selected portable model.
Use `checkpoints/last.ckpt` for completed-epoch resume, and keep `best.ckpt`
for the selected model; they serve different purposes. Keep these artifacts
outside Git. Local records work offline. `wandb = true` optionally adds native W&B logging to
`kdorman90-ucla/kelpseg` (install `wandb` in a runtime-only environment first).

### Prediction panels and regional metrics

With `wandb = true`, `val/predictions` shows the same selected validation chips
each epoch: one kelp-positive chip per represented region where available
(otherwise that region's first chip), plus up to two additional background-only
chips. Selection is deterministic by source/window identity, not model errors.
`train/predictions` shows four distinct chips from the first encountered shuffled,
augmented training batches each epoch. Both reuse predictions from the ordinary
passes; training uses current weights and validation uses validation/EMA weights.
There is no extra inference pass. Batch limits can omit selected validation images.

In W&B, open these image panels and toggle the `label` and `prediction` mask
layers. Masks use threshold .5; class 2 is ignored, including nodata/padding and
augmentation-masked pixels. Captions identify epoch, region, source/date and
original source window; training imagery and masks show the transformed canvas.
RGB displays use red/green/blue when available, otherwise NIR/red/green false
color. Each image uses a 2–98 percentile display stretch; these display colors
are not model inputs. All panel images retain the common model canvas; loading padding is gray and
marked ignored so partial edge chips display correctly in W&B image lists.
Media uploads use the existing W&B run and logger; offline runs retain media
for later W&B synchronization. No media are built when W&B logging is disabled.

`val_region_001/iou` (and each represented region) logs pooled epoch metrics,
including precision, recall, F1, AUROC, calibration error and `pixels`, to both CSV and W&B.
These use the same threshold/ignored pixels as global validation, including
background-only chips. Zero-denominator scores are zero; inspect `pixels` for
support and avoid interpreting empty foreground IoU as evidence of poor quality.
Only validation regions appear; training-only BC regions do not acquire validation
scores. Global `val/iou` remains the checkpoint-selection metric.


## DINOv3 false-color comparison

The DINOv3 comparison now runs four experiments with fresh SegFormer decoders
and binary heads:

| Encoder | False color (NIR/red/green) | Native eight bands |
|---|---|---|
| ViT-S/16 | [recipe](../../configs/train/dinov3/vits16.toml) | [recipe](../../configs/train/dinov3/vits16_planet8.toml) |
| ConvNeXt-Tiny | [recipe](../../configs/train/dinov3/convnext_tiny.toml) | [recipe](../../configs/train/dinov3/convnext_tiny_planet8.toml) |

Both spectral variants use the existing native eight-band chips and identical
source assignments. The encoders were pretrained on RGB web imagery. False color
reassigns the three kernels positionally to NIR/red/green. Eight-band inputs repeat
the RGB kernels cyclically and scale them by 3/8, following our B0 adaptation
policy. This is not multispectral pretraining; all encoder weights are fine-tuned.

All four retain the approved per-band raw means/stds and cap of 5000. These are
Planet normalized inputs, not percentile-stretched pictures or the checkpoint's
RGB display normalization. Bundles store exact ordered bands and preprocessing.

Use `encoder_weights = "dinov3"` with `dinov3_vits16` or
`dinov3_convnext_tiny`; `"none"` builds an offline random model. Supported input
counts are 3 and 8. ImageNet/B0 segmentation weight options cannot be combined with
these encoders. Sources and revisions are pinned in `dinov3.py` and recorded in
run/bundle metadata. The initial download uses the standard Hugging Face cache;
exported bundles and completed-epoch resume reconstruct offline. Sources retain
the [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE).

The decoder peak LR is 0.00015 and floor 0.000003. `encoder_lr_scale = 0.1`
scales both for the encoder (0.000015 and 0.0000003); existing recipes default
to 1. All four use batch 12 and accumulation 2 (effective 24), 70 epochs maximum,
patience 20, seed 42, BCE/Dice, with-BC assignments and validation-only selection.
These match B0's physical batch and accumulation. Bounded RTX 3090 probes at
512 pixels with bf16 mixed precision passed for all four: peak PyTorch reserved
memory was 6.08/6.96 GiB for false-color ViT/ConvNeXt and 6.18/7.04 GiB for
eight-band ViT/ConvNeXt. Checks included three AdamW updates, EMA copies/updates
and validation. Synthetic inputs establish memory fit, not model quality or
full-run completion time.

Run the read-only preflight, then launch in an existing tmux shell when ready:

```bash
cd /home/sky/kelpseg
bash run_dinov3_comparison.sh --check
bash run_dinov3_comparison.sh
```

The [runner](../../run_dinov3_comparison.sh) trains all four in separate processes,
then generates fresh float32 validation predictions and HTML reports for the
four candidates and **both existing B0 references**. It does not retrain B0:

- Eight-band: `planet8-cabc-size-b0-bparams-70ep-20261106T085748896218Z`,
  selected epoch 16 EMA.
- False color: [`planet3-cabc-b0-spectral-037b-70ep-20261110T195653341372Z`](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet3-cabc-b0-spectral-037b-70ep-20261110T195653341372Z),
  selected epoch 46 EMA.

`BASELINE_RUN` and `FALSE_COLOR_RUN` may relocate those same fixed reference
runs. `comparison.md`/CSV and six HTML reports go under the configured external
data root's `comparisons/` directory; an optional absolute positional path selects
a fresh directory. Terminal progress remains direct, without `tee`. A failed step
stops the sequence; partial work is not reused.

The report requires matching validation windows and scored support. Compare
each candidate with the same-band B0 reference and compare spectral variants
within each DINOv3 encoder. Differences from B0 still include capacity, initialization
and encoder LR; they do not isolate pretraining alone. All scores are
validation diagnostics; test data remain held out. Neither baseline is replaced
automatically. Historical full-TIFF spectral scores use a different coverage
scope; this runner recomputes all six on matching disjoint validation chips.

## Encoder size and retained epoch checkpoints

`encoder_name` accepts `mit_b0`, `mit_b1`, `mit_b2`, `mit_b3` (generic default),
`dinov3_vits16`, and `dinov3_convnext_tiny`. See the
[DINOv3 experiment](#dinov3-false-color-comparison) for its three- and eight-band inputs.
The selected size is saved with settings, checkpoints and portable models;
changing it requires a fresh training run. Bundles restore their recorded size
offline.

`checkpoint_epochs = [4, 5, 6]` additionally saves full `epoch-04.ckpt` through
`epoch-06.ckpt` after those epochs finish. Values are one-based completed epochs;
checkpoint internals remain zero-based. The default empty list keeps best/last
only. Snapshots retain validation/EMA weights and current training/optimizer
state and can be exported as bundles. Use `last.ckpt` for supported training
resume. Extra checkpoints can require substantial storage.

## Early stopping

`early_stopping_patience = 10` stops after ten validation checks without a new
best `val/iou`; `early_stopping_min_delta = 0.0` accepts any strict improvement.
The default patience is 0 (disabled). The historical loss-screen
[`losses/bce_dice.toml`](../../configs/train/losses/bce_dice.toml) uses 10; the
selected [`model_size/b0.toml`](../../configs/train/model_size/b0.toml) uses 20.
Keep `epochs = 70`, warmup and LR settings unchanged: early stopping truncates
execution, not the planned cosine schedule. Validation runs once per epoch.
The best checkpoint is still exported, and `run.json` records `stopped_early`.
Resume permits increasing an existing positive patience, while retaining the
saved best score and elapsed wait count. Other stopping/scientific settings must
match. For a completed early-stopped run, the previous model, validation report
and run/settings JSON move into `completed-patience-<old patience>/` only when
resumed fitting starts. The best/last checkpoints continue normally; this archive
preserves selected inference outputs, not a second recovery checkpoint. Dry-run
leaves all artifacts untouched. CSV continues in a separate resume directory and
W&B retains the run ID. See [Lightning early stopping](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html).

## Copy and use a portable model

Copy the complete `model/` directory (`model.json` and `weights.pt`) anywhere
with the installed package. It stores the architecture, ordered bands, cap,
normalization, padding/nodata policy, threshold and selected current/EMA identity.
No recipe, label, training registry, W&B connection or pretrained download is
needed to load it:

```python
from kelpseg.model_bundle import load_model_bundle

bundle = load_model_bundle("/path/to/copied/model")
# image: uint16 H x W x native channels, raw units; declare every native channel.
probabilities = bundle.predict_probabilities(image, source_band_names)
```

The returned float32 array has the image's original height and width, with NaN
at native all-band-zero nodata. Supply an image no larger than 512 × 512;
bottom/right padding is removed from the result. Native validity is determined
before selecting model bands. This helper does not tile images, read labels,
write georeferenced rasters or threshold probabilities. The saved threshold is .5.

To export an existing selected training checkpoint separately:

```python
from kelpseg.model_bundle import export_model_bundle
from kelpseg.training.settings import load_settings

export_model_bundle(
    "/path/to/run/checkpoints/best.ckpt",
    "/path/to/new-model-directory",
    load_settings("/path/to/recipe.toml"),
    run_id="the-saved-run-id",
)
```

Export refuses an existing destination. Loading a bundle restores model weights
only; it does not resume training or restore an optimizer.

## Resume after a completed epoch

For runs started with the completed-epoch support, resume from the last saved
epoch using the same recipe and **same total epoch budget**:

```bash
uv run kelpseg train --recipe /path/to/recipe.toml --resume /path/to/run/checkpoints/last.ckpt --dry-run
uv run kelpseg train --recipe /path/to/recipe.toml --resume /path/to/run/checkpoints/last.ckpt
```

Resume writes a separate `metrics/resume-<UTC timestamp>/metrics.csv`, preserving
earlier metrics; the run record points to the latest CSV.

The command preserves the saved run ID, ordinary model weights, optimizer,
scheduler, progress, random/sampling state and delayed EMA. Work after that
completed epoch is discarded; there is no partial-epoch or mid-batch recovery.
Use `last.ckpt`, not the validation-selected `best.ckpt`. Older Task 033E
checkpoints lack the new resume state and cannot provide exact resume.

Completed-epoch continuation supports zero or multiple loader workers. With
workers, augmentation streams restart and sample order may differ; saved model,
optimizer, scheduler and EMA state still restore. Exact replay requires
`num_workers = 0` and `deterministic = true` from the start. Set
`deterministic = false` to allow faster GPU kernels with numerical variation.
Old multiworker checkpoints without resume metadata remain unsupported.
Changing the recipe (including worker count or deterministic mode), total schedule,
bands or selected memberships fails before training, except for the supported
positive-patience increase described above. Machine roots may relocate:
update `paths_file`, `chips` and `assignments` to the same copied data and retain
the saved run folder under the new `runs_root`. These path changes do not grant
permission to change data identity. CPU equivalence and bounded deterministic CUDA continuation have been checked;
see Task 034 for the device and tolerance. Multiworker continuation is checked
for restored state/progress, not equality of subsequent weights.

## Tiny v2: installed training walkthrough

The [quickstart](../quickstart.md) is the sequential operating example. To check
an installation outside the checkout, complete its preparation, then build a
wheel and install only locked runtime dependencies in a fresh external folder:

```bash
(
  set -e
  work="$HOME/kelpseg-installed-smoke-01"
  test ! -e "$work"
  mkdir -p "$work/runs"
  uv build --out-dir "$work/distributions"
  uv export --frozen --no-default-groups --no-emit-project --output-file "$work/runtime.txt"
  uv venv --python 3.12 "$work/environment"
  uv pip sync --python "$work/environment/bin/python" --require-hashes "$work/runtime.txt"
  uv pip install --python "$work/environment/bin/python" --no-deps "$work"/distributions/*.whl
  cp configs/train/tiny_cpu.toml "$work/tiny.toml"
)
```

Create `paths.toml` in that folder: set `data_root` to the quickstart artifact
root and `runs_root` to this new folder's `runs/`. Set the copied recipe's
`paths_file = "paths.toml"`; retain its CPU limits, zero workers and two-epoch
budget. From that folder use `./environment/bin/kelpseg` in place of
`uv run --frozen kelpseg` for the quickstart's training, prediction and reporting.
The chip/assignment paths must match the prepared data; no further Box fetch
is needed. Runtime installation needs network access before offline use.

For spectral smoke checks, copy the tiny recipe and replace `band_names`, `mean`
and `std` together with values from the desired spectral recipe. Keep CPU limits,
choose `base` or `augmented` explicitly, and give each run a distinct `name`.
Run sequentially. Tiny v2 cannot supply the kelp-rich showcase windows required
by augmentation galleries; use the full training collection for those.

Completed-epoch resume is described above. A successfully completed two-epoch
run cannot be extended by changing its epoch budget. The deliberate interruption
and copied-bundle checks from the earlier installation rehearsal remain in
[Task 033G](../../tasks/033g_tiny_walkthrough_and_handoff.md#outcome-2026-11-03).
