# Task 038B: Use kelpseg in current repository instructions

Status: Complete (including post-rename verification).

## Abstract, inputs and plan

The user chose `kelpseg` for all current naming. Remove the old checkout name
from current operating examples in `docs/gpu_setup.md`,
`docs/guides/diagnostics.md` and `docs/reference/ca_bc_release.md`.
Use the repository's current SSH URL copied from GitHub when cloning into a
folder named `kelpseg`, so the instructions work before or after a GitHub rename.
Preserve recorded historical commands, machine paths and CI links as evidence.

## User decisions required

None for text cleanup. Renaming the actual GitHub repository or moving this
checkout is outside this documentation edit. Package/import/command names
already use `kelpseg`.

## Acceptance and validation

Current examples use `kelpseg`; no application behavior, artifact identity or
historical evidence changes. Inspect the diff, check remaining old-name matches,
verify changed relative links, run repository hygiene and `git diff --check`.
No commit or push unless requested. The next roadmap task remains 039.

## Outcome

Completed 2026-11-12. Updated the three operating documents named above and
the queue/task index. Current checkout examples use `kelpseg`; cloning accepts
the actual SSH URL from GitHub and explicitly names the destination. No
speculative renamed GitHub URL is presented as live.

Validation: all 18 repository hygiene tests passed; `git diff --check` passed.
Reviewed the documentation diff and verified the new task links resolve.
The ten remaining old-name occurrences belong to historical task records and
historical GPU recipes (recorded paths, commands and a CI run URL).
No current operating guide retains the old checkout name.

No code, external artifacts, local folders or GitHub settings changed. The user
subsequently authorized committing these documentation changes; no push was
requested. Exact next action: rename the GitHub repository and update both local
checkouts when ready. The roadmap remains 039 then 042.

## Post-rename verification — 2026-11-12

Verified checkout `/home/sky/kelpseg` at `b133baf` after the user renamed it.
The checkout was initially clean. Origin uses
`git@github.com:kyle-dorman/kelpseg.git`; `git ls-remote origin HEAD` succeeds
and returns the same commit. The installed command's interpreter points into
the renamed checkout. All ten `hakai-ml-train` references remain historical
evidence; none occurs in active code or operating instructions.

Validation:

- `uv sync --python 3.12 --frozen` and `uv lock --check` passed.
- `uv run --frozen kelpseg --version` reports `kelpseg 0.1.0`;
  `uv run --frozen python -m kelpseg --help` passed.
- Full CPU suite: **975 passed**, 92 warnings, in 134.41 seconds, with
  W&B offline, Hugging Face offline, CUDA hidden and uv offline. This includes
  repository hygiene, wheel/source-distribution builds and installation outside
  the checkout, preparation, training, resume, portable models and prediction.
  Warnings concern fixture georeferencing, dependency deprecations, worker/logging
  settings, existing fixture checkpoint directories and multiprocessing.
- Ruff format and lint checks passed (163 Python files formatted).
- Final Markdown link inspection and `git diff --check` passed.

No rename-related failure or code change was needed. Only this record,
`tasks/README.md` and `docs/todo.md` changed. No durable external artifacts were
created; no commits or pushes were made. GPU production training, external
datasets/services and the other local checkout were not exercised.
Exact next action: select and bound Task 039 in a separate window; Task 042
remains after it.
