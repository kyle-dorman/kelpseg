# Task 038E: ADE20K B0 initialization comparison

Status: Complete

## Abstract

Prepare two matched B0 initialization experiments from the same ADE20K-trained
checkpoint: encoder only with a fresh decoder, and encoder plus pretrained
decoder. Both retain a fresh binary kelp head and the existing SMP architecture.
Provide a separate sequential Bash runner with matching validation reports. Do
not start these experiments or interfere with the running OpenEarthMap work.

## Inputs and outputs

- Pinned input: `nvidia/segformer-b0-finetuned-ade-512-512`, revision
  `489d5cd81a0b59fab9b7ea758d3548ebe99677da`, `model.safetensors`.
- Control: selected ImageNet B0 epoch-16 EMA run, `configs/train/model_size/b0.toml`.
- Outputs: two explicit initialization options/recipes, a new Bash runner,
  focused CPU checks, documentation and recorded metadata.

## Implementation plan and contract

- Add `ade20k.py` with pinned download and explicit encoder/full transfer modes.
  Share ordinary HF-B0 conversion/input-adaptation functions with OpenEarthMap
  in `segformer_weights.py`; preserve existing OpenEarthMap behavior.
- Put initialization descriptors in `initialization.py`, recording repo, revision
  and transferred components; keep run/bundle records and offline reload behavior.
- Options `ade20k_b0_encoder` and `ade20k_b0_full` require B0 and disabled
  ImageNet loading. Encoder-only must preserve the initial decoder and head;
  full transfers all decoder parameters/buffers, discarding 150-class output.
- Both recipes in `configs/train/initialization/` match the selected baseline
  outside name/initialization, including eight bands and native normalization.
- `run_ade20k_comparison.sh` runs both fits in separate processes, then makes
  fresh validation reports for baseline/encoder/full and matched-support deltas.
  Inherit the terminal directly, no tee. Stop on first failure; no partial reuse.
  Support read-only --check. Do not alter the existing OpenEarthMap runner.
- Update training guide, experiments/architecture and queue/task navigation.
  No framework, Transformers dependency, production launch or new test scope.

## User decisions required

None. Both variants use the same ADE20K B0 checkpoint, initialization-only
comparison, fresh optimizer/scheduler/EMA and matched training settings. Current
selected baseline and running OpenEarthMap experiment remain unchanged.

## Acceptance and validation

Offline fixtures verify encoder/full transfer, fresh components, input adaptation,
settings/recipe parity, recorded components, and bundle/resume download avoidance.
Run focused existing model/initialization/training checks plus Ruff and repository
hygiene. Inspect the real pinned checkpoint on bounded CPU forward/backward and
bundle reload checks without fitting. Validate Bash syntax, real metadata-only
--check, and stubbed sequencing/failure propagation. Verify documentation links
and git diff --check. No GPU fitting/prediction/reporting launched by the agent.

## Outcome

Completed 2026-11-13. Prepared both options, matched recipes and the separate
runner. No ADE20K training, production prediction/reporting, W&B run, commit or
push was launched. The current OpenEarthMap run and its Bash runner were left
alone; no conclusions about its quality were inferred from partial progress.

### Changes

- `src/kelpseg/ade20k.py` loads the same pinned NVIDIA B0 checkpoint for both
  modes, discarding its 150-class classifier. Encoder-only leaves all fresh
  decoder parameters/buffers intact; full transfers decoder weights/buffers.
- `segformer_weights.py` holds the previously verified HF B0 conversion and RGB
  adaptation, shared with OpenEarthMap. `openearthmap.py` retains its pinned
  source and full-transfer behavior; regression checks pass.
- `initialization.py` describes all three optional segmentation-trained
  initializers. `models.py` accepts the two new names; `training/run.py` and
  `model_bundle.py` use the shared descriptive metadata. Existing setting and
  resume paths already pass/clear model_weights as required.
- `configs/train/initialization/ade20k_b0_encoder.toml` and
  `ade20k_b0_full.toml` differ from the selected B0 control only in initialization
  and distinct names. No training policy, normalization or assignment changes.
- `run_ade20k_comparison.sh` trains encoder-only then full in separate processes;
  then produces three fresh matching validation reports and CSV/Markdown deltas
  against ImageNet and between the two ADE20K variants. Stdout/stderr go directly
  to the terminal. --check is read-only; first failure stops subsequent steps.
- `tests/test_ade20k.py` checks the two modes and `test_training_run.py` extends
  preview/resume coverage. Updated training guide, experiments/architecture,
  queue and task navigation. All earlier user/uncommitted changes preserved.

### Validation

- 13 ADE20K tests passed, including exact same-seed encoder equality, preserved
  fresh decoder/head, full decoder transfer, all 3/4/8 input options, pinned
  source, recipe parity, incompatibility errors and offline bundle predictions.
- 171 regression checks passed across OpenEarthMap, settings/model, training
  runner/resume, portable bundles and repository hygiene. No preexisting
  unrelated failures were encountered. The tests use bounded synthetic fixtures,
  not production training.
- Real pinned checkpoint: both modes passed CPU `1x8x512x512` forward/backward
  checks with the configured BCE+Dice loss and ignored pixels. Every trainable
  parameter had finite gradients. Initial encoders matched exactly. Both exported
  bundles reproduced outputs exactly. No optimizer step or Trainer fit occurred
  in these manual checks. Temporary artifacts:
  `/tmp/kelpseg-ade20k-check-9n31kpe0`.
- The new runner's real --check passed for both recipes and the saved baseline:
  8,518 eligible training chips, 933 validation chips, 1,262 test metadata-only
  rows; each recipe retains at most 70 epochs / 24,850 planned optimizer updates,
  early stopping patience 20 and effective batch 24. No output folder was made.
  Disposable log: `/tmp/kelpseg-ade20k-runner-check.log`.
- Bash syntax/help and all embedded Python Ruff checks passed. Stubbed actual
  runner execution verified success order, encoder/full training failures,
  encoder-report failure, --check without outputs and all three IoU delta columns.
  Disposable evidence: `/tmp/ade-runner-stub-ksofrzns`.
- Repository Ruff format/check, hygiene, changed-document relative links and
  `git diff --check` passed. Focused logs are under `/tmp/kelpseg-ade20k-*`.

### Artifacts, limits and simplicity

Only the pinned ADE20K source was added to the external standard Hugging Face
cache. No durable trained model or comparison result was created. CPU compatibility
checks do not establish kelp quality or production CUDA throughput. The source
weights are transferred into existing SMP numerical conventions, not a bitwise
copy of the Hugging Face runtime. Reports compare validation-selected models on
validation data; test payloads stay excluded.

Routine use is two explicit recipes and one Bash invocation. Shared conversion
avoids duplicate weight mappings; there is no Transformers runtime, new dependency,
framework, checksum/receipt system or partial-work reuse mechanism. Existing
OpenEarthMap/default behavior remains available while the user's run finishes.

### Exact next action

After the current run finishes, the user may execute
`bash /home/sky/kelpseg/run_ade20k_comparison.sh` inside their existing tmux shell.
The script is not running or scheduled and does not automatically wait for other
jobs. Review comparison.md and the three HTML reports after completion. Keep
held-out test results out of recipe/checkpoint selection. The next development
roadmap task remains 039 in its own window, followed by 042.
