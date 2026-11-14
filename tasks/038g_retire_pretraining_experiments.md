# Task 038G: Summarize and retire pretraining/encoder experiments

Status: Complete.

## Scope and authorization

User requests a summary of OpenEarthMap B0, ADE20K encoder/decoder variants,
and DINOv3 experiments, followed by a preservation commit and removal of their
experimental code. Commit the archive first, then commit the cleanup while
retaining the summary and recovery reference. Do not push or remove external
runs, checkpoints, reports, source data or unrelated local edits.

## Plan and expected files

Write `docs/reference/pretraining_experiments.md` from saved run/report evidence,
including exact identities, evaluation scopes, limitations and conclusions.
Archive experimental adapters, initialization converters, recipes, runners,
relevant tests and task records. Then remove those files and restore shared
model/training code to the pre-experiment baseline where the entire diff belongs
to this work. Remove experiment-only dependency declarations, retaining locked
transitive requirements. Update active docs and history links; historical task
records retain evidence with archive recovery links. Keep MiT B0/B1/B2/B3 and
normal ImageNet/random initialization. DINOv3 bundle use requires archived code.

## Decisions and checks

No blocking choices. Two commits make preservation and retirement reviewable.
Do not include unrelated Task038B rename-verification edits, including portions
of shared queue/navigation files. Validate the archive before commit, then run
focused baseline train/resume/bundle tests, hygiene and Ruff after removal.
Inspect local Markdown links and `git diff --check`. No training launches.

## Outcome

Preservation commit: `c5553dfa96f4f8c1017ed2e06fcb83eeb3152628`.
It contains the complete implementation, seven recipes, three launchers, focused
tests, task records and results summary. Archive checks: 72 experiment/hygiene
tests, Ruff format/lint and diff checks passed.

Retired the OpenEarthMap/ADE20K adapters and shared HF converter, DINOv3 adapter,
initialization descriptors, seven recipes, three launchers and their three test
modules. Restored shared model/bundle/training/settings/scheduler/resume code,
related baseline tests and direct dependency declarations to their pre-experiment
state (`b133baf`). The entire maintained `src/`, `tests/`, pyproject and lockfile
match that baseline; transitive timm/HF/safetensors dependencies remain where SMP
requires them. Standard B0/B1/B2/B3 and ImageNet/random initialization remain.

Retained the results summary and historical Tasks 038C–038F with explicit archive
notices. Updated product, training, experiment, history, index and queue docs to
remove current instructions for retired features. No external source data, runs,
checkpoints, pretrained caches or reports were deleted. The summary distinguishes
the incomplete OpenEarthMap evidence from completed float32 ADE20K/DINOv3 reports.

Real selected eight-band B0 and both ADE20K B0 bundles load under the cleaned
code and produce finite 1x1x64x64 logits. Local Markdown targets and diff checks
pass. Full CPU suite: **975 passed**, 92 ordinary fixture/dependency warnings,
in 128.75 seconds, with CUDA hidden, HF offline and W&B offline. No unrelated
failures. Log: `/tmp/kelpseg-retirement-tests.log`. Ruff format/lint and
`uv lock --check` also passed. The cleanup commit is the commit containing this
completed outcome; the exact preservation commit above remains the recovery point.
Unrelated Task038B rename-verification edits in its task file, queue paragraph
and navigation row are excluded from both commits and preserved in the worktree.

No training or push was launched. After review, the user may push both commits.
The next roadmap task remains 039 (new-region recipes), then 042, in a separate
window; no implementation from those tasks was started.

## Recovery

Full experiment implementation and summary are preserved at
`c5553dfa96f4f8c1017ed2e06fcb83eeb3152628`. Inspect without modifying this checkout:

```bash
git show c5553df:src/kelpseg/dinov3.py
git show c5553df:run_dinov3_comparison.sh
```

Export a separate archive for historical model use (choose a fresh destination):

```bash
mkdir /tmp/kelpseg-pretraining-archive
git archive c5553df | tar -x -C /tmp/kelpseg-pretraining-archive
cd /tmp/kelpseg-pretraining-archive
uv sync --python 3.12 --frozen
```

Supply the appropriate external model/data paths explicitly. Do not restore
retired modules into the active package. DINOv3 bundles need the archived model
builder. ADE20K B0 bundles have the standard SMP MiT structure and remain usable
for inference in the maintained toolkit; retired recipe initialization and
experiment resume require the archive. No checkpoints or reports are stored in
Git; local evidence paths are in the retained summary. Neither commit is pushed.
