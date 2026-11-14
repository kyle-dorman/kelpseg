# Task 038C: OpenEarthMap B0 initialization

Status: Complete

> Historical implementation record. Experiment code/recipes were removed after
> preservation in commit `c5553df`. Commands below describe the archived version,
> not the maintained checkout. See [results](../docs/reference/pretraining_experiments.md)
> and [recovery](038g_retire_pretraining_experiments.md#recovery).


## Abstract

Add a bounded, opt-in initialization experiment using the published OpenEarthMap
B0 encoder and decoder in the existing SMP architecture. Preserve the current
selected baseline. This task implements and checks the option; it does not launch
training or begin the new-region recipes in Task 039.

## Inputs and outputs

- Input checkpoint: `boehnen/satlens-segformer`, revision
  `19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6`, `model.safetensors` (MIT).
- Reference: `configs/train/model_size/b0.toml`, selected with-BC BCE+Dice recipe.
- Outputs: explicit model initialization option, matched comparison recipe,
  offline focused tests, and researcher instructions.

## Implementation plan and contract

- Add `src/kelpseg/openearthmap.py`: plain weight conversion/loading functions;
  download the pinned checkpoint through the ordinary Hugging Face cache.
- Extend `models.py` and training settings/module/run with
  `model_weights = "openearthmap_b0"`. Require B0 and `encoder_weights = "none"`;
  retain the existing default and forbid ambiguous mixed initialization.
- Map encoder and decoder weights, combine attention K/V, reverse decoder
  projection indices, discard the nine-class classifier, and preserve our fresh
  binary head. Adapt RGB kernels using the existing cyclic 3/channel-count rule.
- Preserve SMP architecture, normalization, loss, splits, augmentation and EMA.
  Record source and revision in run/bundle metadata. Dry-run and bundle loading
  must not download; completed-epoch resume must not reinitialize weights.
- Add `configs/train/initialization/openearthmap_b0.toml`, changing only run name
  and initialization relative to the selected B0 reference.
- Document usage in `docs/guides/training.md`, initialization in
  `docs/experiments.md`, and the architecture boundary in `docs/architecture.md`.
  Declare direct download/serialization dependencies in the locked environment.
- Keep model artifacts outside Git. No new command or framework is needed:
  existing `kelpseg train --recipe ... --dry-run` previews the comparison.

## User decisions required

None. The user authorized implementation and a matched recipe, and explicitly
forbade launching training. Default to retaining the SMP numerical conventions
rather than reproducing the original Hugging Face model exactly.

## Validation and acceptance

- Offline tensor fixtures check K/V ordering, projection mapping, input adaptation,
  full encoder/decoder coverage and binary-head preservation; reject wrong shapes
  and incompatible options. Check recipe equality outside initialization/name.
- Check initialization metadata, dry-run without downloads, fresh initialization
  and resume/bundle paths without re-downloading. Run relevant existing model,
  settings, run, resume and bundle tests. CPU synthetic checks are permitted;
  no production fitting or experiment launch.
- Inspect the real checkpoint with bounded forward/backward and save/load checks.
- Run Ruff formatting/lint, repository hygiene tests, Markdown link checks, and
  `git diff --check`.
- Routine use must require only the recipe, no Python edits. No source imagery or
  user edits are changed. No commits or pushes.

## Non-goals

Training, score claims, generic model-zoo support, new-region fitting, exact HF
numerical reproduction, new recovery machinery, and changing the selected model.

## Outcome

Completed 2026-11-13. No production training or experiment was launched. No
commit or push was made.

### Changes

- `src/kelpseg/openearthmap.py` converts the pinned HF checkpoint into the
  existing SMP B0 model, preserving the new binary head. It imports all 190
  encoder/decoder parameter and buffer tensors, joins K/V projections and
  reverses the decoder projection numbering. All source/target entries and
  tensor shapes must match; incomplete transfer fails with an actionable error.
- `models.py`, `training/settings.py` and `training/module.py` expose validated
  `model_weights="openearthmap_b0"` initialization, requiring B0 with no competing
  ImageNet initializer. RGB/4/8-band input support follows the existing policy.
- `training/run.py` and `model_bundle.py` record the pinned source/revision and
  transfer policy. Preview remains metadata-only. Resume disables both initial
  weight sources; exported models rebuild offline in the existing architecture.
  `training/resume.py` omits the unset optional initializer from ordinary
  baseline identities so this addition does not invalidate existing baseline
  checkpoint identities. Explicit OpenEarthMap identities retain the option.
- `configs/train/initialization/openearthmap_b0.toml` matches the selected
  `model_size/b0.toml` recipe except for name and initialization.
- `pyproject.toml` and `uv.lock` declare the already-installed Hugging Face Hub
  and safetensors dependencies directly; no locked package version changed.
- `tests/test_openearthmap.py`, `test_training_run.py`, and
  `test_training_resume.py` cover conversion, settings, matched recipe, portable
  bundles, metadata, preview and initialization-free continuation.
- Researcher instructions live in `docs/guides/training.md`; experiment and
  architecture contracts plus task navigation/queue were updated.
  Preexisting user edits in `tasks/038b_repository_naming.md`, `tasks/README.md`
  and `docs/todo.md` were preserved.

### Validation and evidence

- Focused suite: 163 passing checks across OpenEarthMap, training settings/model,
  run/resume, portable bundle and repository hygiene tests. After adding two
  access-error checks and explicit default-identity coverage, the affected
  OpenEarthMap/resume subset passed all 25 checks. No unrelated failures found.
  Tests use offline fixtures; existing bounded synthetic training tests are not
  production runs.
- Real pinned checkpoint: eight-band `1x8x512x512` CPU forward/backward passed,
  including finite gradients for every trainable parameter. A synthetic ignored
  label region used the configured BCE+Dice objective. No optimizer step or
  Trainer fit was executed in this manual check. Exported/reloaded model outputs
  were exactly equal; source revision and transfer metadata survived export.
- Both production recipes passed installed CLI `--dry-run` with
  `HF_HUB_OFFLINE=1`. Counts, paths, loss, augmentation, selection, EMA and budget
  were identical: 8,518 eligible training windows, 933 validation windows,
  1,262 test metadata rows, 710 training batches/epoch, effective batch 24 and
  24,850 planned optimizer updates over at most 70 epochs. Early stopping stays
  enabled with patience 20. Neither preview created a run directory.
- Ruff format/check passed. Repository hygiene, changed-document relative links,
  and `git diff --check` passed. Independent read-only review found no correctness
  blocker; its input-file error-message suggestion was addressed.

### External artifacts and limitations

The pinned source is cached in the ordinary Hugging Face cache outside Git.
Manual synthetic artifacts are temporary at
`/tmp/kelpseg-oem-implemented-a8k27ks2`; preview/test logs under
`/tmp/kelpseg-openearthmap-*` and `/tmp/kelpseg-imagenet-preview.json` are disposable.
No durable trained model, run result, Box backup or W&B experiment was created.

This transfers weights into SMP, retaining its numerical/regularization and
upsampling conventions; it does not reproduce HF logits exactly. Planet
normalization and positional RGB kernel expansion are unchanged. No kelp
performance, CUDA memory or throughput claim is established. First use on a
machine requires access to the pinned checkpoint or its existing HF cache.

### Simplicity and next action

Routine use is a single explicit recipe; no Python edits, generic model zoo,
new CLI command, checksum/receipt machinery or Transformers dependency. Existing
initialization remains the default; no obsolete framework was introduced or
retained for this experiment.

Exact next action: review the uncommitted changes. If the user later authorizes
training, launch the documented OpenEarthMap recipe as a new run and compare its
validation-selected result with the matched ImageNet baseline, keeping test
results held out during selection. Do not launch it as part of this task.
The roadmap's next development task remains 039 (new-region recipes), to be
bounded in its own window; 042 follows.
