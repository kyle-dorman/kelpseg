# Task 038D: OpenEarthMap comparison Bash runner

Status: Complete

## Abstract and boundaries

Provide the user-requested Bash script and an existing-tmux launch command for
Task 038C's initialization experiment. Prepare and validate the runner without
launching fitting, prediction or production reporting in this task. Keep 039
and 042 pending.

## Inputs and outputs

Inputs: the 038C recipe, paths.local.toml, canonical chips/assignments, and the
selected B0 epoch-16 EMA run documented in docs/experiments.md. Outputs: root-level
`run_openearthmap_comparison.sh`, usage in the training guide, and this record.
Generated runs/logs/predictions/reports stay outside Git.

## Plan and implementation contract

- Bash runs steps in sequence with `set -euo pipefail`; first failure stops it.
- `--check` validates input paths, selected baseline identity, recipe parity and
  GPU availability without fitting, model downloads or output creation.
- Resolve machine data roots from paths.local.toml; allow BASELINE_RUN only for
  relocating the same selected baseline. Refuse existing comparison output.
- Train the new recipe through the installed training function, recording its
  returned exact output directory rather than guessing a newest run.
- Use installed predict-chips/report commands on both models with validation
  scope and fresh outputs. Do not modify baseline files or touch test payloads.
- Compare exact selected window identities and scored support before writing
  pooled metrics/deltas to comparison.csv and a small comparison.md with links.
- Keep report options identical and send phase output directly to the terminal.
  No new framework, recovery workflow, scripts/ directory or package command.

## User decisions required

None. Default to all validation chips and the current selected baseline, with
ordinary reports and 12 diagnostic examples. User will launch inside tmux.

## Acceptance and validation

Bash syntax, read-only --check against real inputs, bounded stub execution for
sequence/failure propagation, Ruff checks, repository hygiene, Markdown links
and git diff --check. Confirm the launch command works from an existing shell
without creating a new tmux session. No training launched by the agent.

## Outcome

Completed 2026-11-13. Added executable `run_openearthmap_comparison.sh` at the
repository root and its usage under `docs/guides/training.md`. Updated only the
current task navigation/queue in addition to those files. Preserved all existing
038C and other uncommitted changes. No commits or pushes.

The script uses the current baseline at
`/home/sky/data/model-size-runs/planet8-cabc-size-b0-bparams-70ep-20261106T085748896218Z`,
resolved from the configured data root by default. Its saved settings match the
OpenEarthMap recipe outside initialization/name/machine-path reference. The
baseline bundle is epoch 16 EMA at threshold 0.5. It trains only the candidate,
then writes fresh float32 validation predictions/reports for both, followed by
pooled CSV/Markdown metrics and differences. Exact window identities, coverage,
threshold and labeled kelp support must match before summary publication.

Validation:

- `bash -n` and `--help` passed.
- Real `--check` passed: saved settings match, CUDA is available, 8,518 eligible
  training windows and 933 validation windows; no output directory was created.
  Evidence: disposable `/tmp/kelpseg-oem-runner-check.log`.
- A temporary stand-in `uv` executable exercised the actual shell runner with
  paths containing spaces. The success case ran preflight, training, baseline
  prediction/report, candidate prediction/report and the real inline comparison
  formatter in order. Forced training failure and baseline-report failure each
  returned exit 17 and prevented all subsequent phases. Check-only executed just
  preflight and created no output. Evidence:
  `/tmp/oem-runner-stub-67uvbsvz` (disposable synthetic fixtures).
- Embedded Python blocks were formatted/linted separately. Repository Ruff,
  hygiene, relative documentation links and `git diff --check` passed.
- Independent read-only review found no substantive failure-propagation,
  path/CLI or scientific-scope issue.

No training, GPU inference, production reporting, W&B experiment or durable
external artifact was created during this task. Normal runtime checkpoints stay
under runs_root; comparison outputs default to data_root/comparisons with a UTC
suffix. Outputs are not automatically backed up. Existing report CPU/memory
requirements apply. This is a sequential recipe, not a restart/recovery system.

Simplicity: one Bash file, the existing installed training function and CLI
commands, ordinary JSON/TOML/CSV, no new dependency or scripts/ directory.

Exact next action: the user can run
`bash /home/sky/kelpseg/run_openearthmap_comparison.sh` in their existing tmux
shell. The agent has not launched it. Review the generated comparison.md and
both report.html files after completion; keep test data held out during recipe
selection. Development roadmap remains 039, then 042.

### Terminal progress correction — 2026-11-13

At the user's request, removed all three `tee` pipelines and stderr redirections
from the Bash runner. Training, prediction and reporting now inherit the tmux
terminal directly. Updated the training guide to remove console-log promises;
structured training results, CSV metrics and W&B logging remain available.
Bash syntax and `git diff --check` passed. No workload was launched.
The launch command is unchanged. An already-running process retains its existing
output pipes; this edit applies to future script invocations.
