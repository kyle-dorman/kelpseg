# Task 038F: DINOv3 false-color comparison

Status: Complete

## Abstract and scope

Set up DINOv3 ViT-S/16 and ConvNeXt-Tiny encoders with fresh SegFormer
decoders for a user-launched comparison. Use false color NIR/red/green as
explicitly requested. Do not launch production training or change selected baseline.

## Inputs and outputs

Use existing canonical chips, augmented assignments, baseline BCE/Dice policy,
and selected eight-band B0 run. Add two explicit recipes, a model adapter,
portable initialization metadata, focused tests, and a sequential Bash runner.

## Implementation plan

- Add `src/kelpseg/dinov3.py` using installed timm encoders and SMP decoder.
- Extend model/settings/initialization and offline bundle reconstruction.
- Use four intermediate ViT blocks; ConvNeXt supplies four native scales.
- Keep approved per-band raw normalization; record false-color spectral remapping.
- Add `configs/train/dinov3/` recipes and `run_dinov3_comparison.sh`.
- Preserve effective batch 24 with smaller physical batches, lower encoder LR,
  identical assignments/loss and validation-only selection/reporting.
- Runner uses direct terminal output, stops on failures and refuses old outputs.

## User decisions required

None blocking. User selected both encoders and false color (existing planet3:
NIR/red/green). Default fresh decoder, full encoder fine tuning, no launch.
Eight-band B0 is a practical reference, not a controlled encoder-only comparison.

## Checks and acceptance

Verify offline shape/gradient and bundle reconstruction, settings rejection,
initialization provenance, resumed/offline construction, runner preflight and
matched report support. Run focused tests, ruff format/check, repository hygiene,
and git diff --check. Real pretrained loading and bounded smoke checks may use
external cache; no production training. No new dataset, broad sweep, or recovery
framework. Record files, external artifacts, results, limits and next action.

## Outcome

Implemented both opt-in encoders and the manual comparison runner. The selected
baseline is unchanged. No production training, predictions or reports were run.

### Changed files

- `src/kelpseg/dinov3.py`: pinned timm encoders, four feature maps, fresh SMP
  decoder and binary head; three-channel input and offline reconstruction.
- `models.py`, `initialization.py`, `model_bundle.py`, training settings/run/module/
  scheduler/resume: supported model choices, source metadata, encoder LR scaling,
  consistent scaled LR floor, preserved default optimizer/resume behavior.
- `pyproject.toml`, `uv.lock`: explicit timm dependency with installed version
  retained.
- `configs/train/dinov3/{vits16,convnext_tiny}.toml` and
  `run_dinov3_comparison.sh`: NIR/red/green, approved raw stats, batch 3 x8,
  encoder LR scale0.1, sequential fits and matched validation reports.
- Existing ADE20K/OpenEarthMap runner preflights default missing historical
  encoder LR scale to1, retaining exact comparison of explicit values.
- `tests/test_dinov3.py` and training-loop fixture: architecture/backprop,
  copied offline bundle parity, invalid settings, matched recipes, source-only
  dry-run, optimizer grouping, LR schedule and scheduler/optimizer restoration.
- Training guide, architecture, product, experiments, task queue/navigation.

### Validation

- 154 distinct focused tests passed across DINOv3, ADE20K, OpenEarthMap,
  bundle, training-run/resume/loop and repository hygiene checks. Initial
  regression invocation passed134 tests; final DINOv3/loop invocation passed27,
  including the7 DINOv3 cases already counted in the first invocation.
- Actual pretrained CPU checks: both models accepted1x3x512x512, returned finite
  1x1x512x512 logits and propagated gradients into encoder and decoder. No fit or
  optimizer step was used in these real-model checks. ViT160 and ConvNeXt178
  encoder tensors had gradients, plus11 decoder tensors each. Initial patch/stem
  weights exactly matched the respective downloaded sources.
- Both copied-bundle tests reproduced logits exactly without source downloads.
- Real DINOv3 runner `--check` passed:8,518 training chips,933 validation,
  1,262 test metadata only;355 updates/epoch,24,850 max updates. Existing two
  runners also passed their preflights after compatibility adjustment.
- Bash syntax, embedded Python compilation, and temporary stub execution verified
  sequential fits, three predictions/reports, comparison files, stop-on-failure
  and rejection of mismatched validation windows.
- Ruff format/check, Markdown local link targets, `git diff --check` and
  repository hygiene passed. Ordinary Lightning worker/checkpoint warnings only.

### External artifacts and limits

Only pretrained sources were added to the standard external HF cache:
`/home/sky/.cache/huggingface/hub/models--timm--vit_small_patch16_dinov3.lvd1689m/snapshots/3bf4720a82ec2066db88137180ff1f83a675cef0/model.safetensors`
and
`/home/sky/.cache/huggingface/hub/models--timm--convnext_tiny.dinov3_lvd1689m/snapshots/39c7bb32f83c11bbb13cb23058f8852a4dac88ed/model.safetensors`.
Both downloaded successfully without an access prompt. Temporary root test logs
are `/tmp/kelpseg-dinov3-tests.log` and
`/tmp/kelpseg-dinov3-final-tests.log`; runner fixture evidence is under
`/tmp/kelpseg-dinov3-runner-oxdcj6fc`.

CPU checks do not establish CUDA production memory/throughput or kelp accuracy.
Both pretrained sources are RGB web-image encoders; false color retains kernels
positionally and learns NIR/red/green through fine tuning with existing Planet
normalization. Relative to eight-band B0, bands, capacity, physical batch and
encoder LR differ. Effective batch matching does not match BatchNorm statistics.
Reports are validation diagnostics, not held-out test performance. No baseline
replacement, pretrained decoder, new preprocessing scheme or broad model zoo.

Routine use remains two explicit recipes and one Bash command. No new framework,
receipt system, partial-work recovery or source-data mutation was introduced.
All changes remain uncommitted for review.

### Exact next action

User may run `bash /home/sky/kelpseg/run_dinov3_comparison.sh` in an existing tmux
shell after other jobs finish. It trains ViT-S then ConvNeXt-Tiny and writes
comparison.md/CSV plus three HTML reports under the configured data root's
comparisons directory. Nothing is running or scheduled by this task. The next
roadmap development task remains039 in its own window, then042.

## Four-run extension (user follow-up)

Status: Complete. Extend this same bounded comparison to both encoders with
native eight bands as well as false color. Adapt pretrained RGB input kernels
by cyclic repetition scaled by 3/8, preserving the baseline's positional policy.
Add the completed false-color B0 run as a second reference and generate six
matched validation reports. Keep both references fixed, no retraining or launch.
Validate pretrained kernel adaptation,8-band gradient/bundle checks, four-recipe
preflight, report support and stop-on-failure. Update usage and record outcome.

### Extension outcome

Added native eight-band variants in `vits16_planet8.toml` and
`convnext_tiny_planet8.toml`. Both encoder adapters, settings and offline bundles
now support three or eight inputs. Metadata records the actual input adaptation.
The runner now executes four fits and six reports, retaining terminal progress
and ordinary failure stopping. No training or prediction was launched.

Verified the old false-color W&B/run ID from local run and bundle records:
`planet3-cabc-b0-spectral-037b-70ep-20261110T195653341372Z`, complete, NIR/red/green,
selected epoch 46 EMA (stored checkpoint_epoch 45), threshold 0.5. Its complete
run lives under `/home/sky/data/runs/`. The runner supports `FALSE_COLOR_RUN`
for relocation alongside `BASELINE_RUN`. Both are fixed references, not retrained.

Validation: 116 focused tests passed (DINOv3, bundles, training-run and hygiene)
in `/tmp/kelpseg-dinov3-eight-final.log`. Added eight-band forward/backward,
copied-bundle, metadata dry-run, unsupported four-band rejection and matched
spectral recipe checks. Both real cached pretrained eight-band models produced
finite 512x512 predictions and encoder/decoder gradients. Every input kernel
slice exactly matched source channel modulo three scaled by 3/8. No new downloads
or optimizer steps were needed for these smoke checks. Ruff format/check,
Markdown local links, Bash syntax and diff whitespace checks passed.

The actual four-recipe preflight passed with both references, each planning
355 updates/epoch and effective batch 24. The stub runner verified four fits,
six reports, stop on failure, matched-window rejection and correct same-band,
spectral and architecture differences. Temporary stub artifacts are under
`/tmp/kelpseg-dinov3-runner-wgsa2ljt`; no new durable experiment artifacts exist.

The exact next action is unchanged: manually execute
`bash /home/sky/kelpseg/run_dinov3_comparison.sh` in tmux. It now runs false-color
ViT-S, false-color ConvNeXt-Tiny, eight-band ViT-S, then eight-band ConvNeXt-Tiny.
Reports are under a fresh `comparisons/dinov3-spectral-<UTC timestamp>` directory
and include both B0 references. GPU completion time is not established. Keep
same-band comparisons distinct from a pure initialization comparison: architecture,
capacity, physical batch and encoder LR also differ from B0. All scores remain
validation diagnostics; the current selected baseline and roadmap stay unchanged.

## Batch-12 GPU memory check (user follow-up)

Status: Complete. Probe all four recipes on the available GPU at physical
batch 12 with accumulation 2, including mixed-precision backward, optimizer
state, EMA and validation. Use bounded synthetic full-size tensors, no full
training or W&B run. Increase all four recipes only if measured headroom is
adequate; preserve effective batch 24. Record peak memory, validation and limits.

### Batch-12 outcome

The idle GPU was an NVIDIA GeForce RTX 3090 (24 GB), with no compute processes.
All four variants passed batch 12 at 512x512 in bf16 mixed precision, using the
actual TrainingModule, BCE/Dice objective, AdamW parameter groups, metrics,
gradient clipping and accumulation 2. Each isolated process ran six synthetic
training microbatches (three optimizer updates) and two validation batches. EMA
was deliberately updated immediately so its GPU copy/update and validation
weight swaps were exercised without waiting 101 updates. AdamW state was present
for all used parameters. No W&B run, checkpoint, portable model, production fit
or source-data mutation occurred. Probe learning schedules were shortened solely
for the bounded check; the production 70-epoch schedule is unchanged.

| Variant | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|
| ViT-S false color | 5.20 | 6.08 |
| ConvNeXt-Tiny false color | 6.45 | 6.96 |
| ViT-S eight bands | 5.30 | 6.18 |
| ConvNeXt-Tiny eight bands | 6.54 | 7.04 |

These are PyTorch memory peaks, not whole-device totals. Synthetic full-size
inputs measure memory fit, not accuracy or end-to-end data throughput. Logging
was disabled; ample remaining device memory covers ordinary logging/diagnostic
work. The GPU returned to its idle allocation after the probes.

Updated all four recipes to `batch_size = 12` and
`accumulate_grad_batches = 2`; effective batch remains 24. The runner preflight
requires this pair and now compares physical batching with the saved B0 settings.
Updated its report wording, training guide, experiment policy and recipe tests
so physical batch is no longer described as a difference from B0.

Evidence: `/home/sky/data/analyses/dinov3-batch12-memory-20261114/` contains
`probe.py` and one JSON/log per variant (directory date follows host UTC).
Actual four-recipe `--check`, four focused recipe tests, Bash syntax, Ruff
format/check, repository hygiene and diff checks passed. Production training
has not started. The exact next action remains the user's manual
`bash /home/sky/kelpseg/run_dinov3_comparison.sh` in tmux; it now uses 12x2 for
all four fits. No new roadmap task was started.
