# Task 038G: Summarize and retire pretraining/encoder experiments

Status: In progress.

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

Pending. Record archive commit, files retired, retained artifacts, validation,
unresolved issues and next action. Roadmap remains039 then042.
