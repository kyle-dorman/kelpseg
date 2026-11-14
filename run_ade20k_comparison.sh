#!/usr/bin/env bash
# Train two ADE20K B0 variants, then compare with the selected ImageNet baseline.
# Usage: bash run_ade20k_comparison.sh [--check] [ABSOLUTE_OUTPUT_DIRECTORY]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
check_only=false
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash run_ade20k_comparison.sh [--check] [ABSOLUTE_OUTPUT_DIRECTORY]"
    echo "BASELINE_RUN may relocate the documented epoch-16 baseline run."
    echo "--check validates inputs without training, downloads, or output creation."
    exit 0
fi
if [[ "${1:-}" == "--check" ]]; then
    check_only=true
    shift
fi
if (( $# > 1 )); then
    echo "Expected at most one output directory; use --help." >&2
    exit 1
fi
encoder_recipe="configs/train/initialization/ade20k_b0_encoder.toml"
full_recipe="configs/train/initialization/ade20k_b0_full.toml"
baseline_id="planet8-cabc-size-b0-bparams-70ep-20261106T085748896218Z"
data_root=$(uv run --frozen python -c 'from kelpseg.paths import load_paths; print(load_paths("paths.local.toml").data_root)')
baseline_run="${BASELINE_RUN:-$data_root/model-size-runs/$baseline_id}"
comparison="${1:-$data_root/comparisons/ade20k-b0-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "$comparison" != /* || -e "$comparison" ]]; then
    echo "Choose a fresh absolute output directory: $comparison" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1

# Read-only preflight: compare actual saved baseline settings and preview scope.
uv run --frozen python - "$encoder_recipe" "$full_recipe" "$baseline_run" "$baseline_id" <<'PY'
import json
import sys
from pathlib import Path

import torch

from kelpseg.paths import check_paths, load_paths
from kelpseg.training.run import run_training
from kelpseg.training.settings import load_settings

encoder_recipe, full_recipe, baseline_path, baseline_id = sys.argv[1:]
baseline = Path(baseline_path)
for relative in ("run.json", "model/model.json", "model/weights.pt"):
    if not (baseline / relative).is_file():
        raise SystemExit(
            f"Missing baseline file: {baseline / relative}. Set BASELINE_RUN to the complete selected run."
        )
saved = json.loads((baseline / "run.json").read_text())
model = json.loads((baseline / "model/model.json").read_text())
if (
    saved["status"] != "complete"
    or saved["run_id"] != baseline_id
    or model["run_id"] != baseline_id
):
    raise SystemExit("Expected the completed, selected ImageNet B0 baseline run.")
if (
    model["checkpoint_epoch"] != 15
    or model["weight_kind"] != "ema"
    or model["threshold"] != 0.5
):
    raise SystemExit("Expected the selected epoch-16 EMA baseline at threshold 0.5.")
previews = []
for recipe, expected_weights in (
    (encoder_recipe, "ade20k_b0_encoder"),
    (full_recipe, "ade20k_b0_full"),
):
    settings = load_settings(recipe)
    if settings.model_weights != expected_weights:
        raise SystemExit(f"Expected {expected_weights} initialization in {recipe}.")
    preview = run_training(
        settings, check_paths(load_paths(settings.paths_file)), dry_run=True
    )
    current = json.loads(json.dumps(preview["settings"]))
    reference = dict(saved["settings"])
    reference.setdefault("encoder_lr_scale", 1.0)
    for key in ("name", "paths_file", "encoder_weights", "model_weights"):
        current.pop(key, None)
        reference.pop(key, None)
    if current != reference:
        changed = sorted(
            key
            for key in current.keys() | reference.keys()
            if current.get(key) != reference.get(key)
        )
        raise SystemExit(
            f"{recipe} differs from the saved baseline beyond initialization: {changed}"
        )
    for key in ("chips", "assignments"):
        if Path(preview[key]).resolve() != Path(saved[key]).resolve():
            raise SystemExit(f"Baseline and candidate must use matching {key} paths.")
    previews.append(
        {
            "recipe": recipe,
            "initialization": preview["initialization"],
            "counts": preview["counts"],
            "budget": preview["budget"],
        }
    )
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable; run this GPU experiment on the configured GPU host."
    )
print(json.dumps({"baseline": str(baseline), "experiments": previews}, indent=2))
PY
if "$check_only"; then
    echo "Checks passed. No training, prediction, or reporting launched. Planned comparison: $comparison"
    exit 0
fi

mkdir -p "$(dirname "$comparison")"
mkdir "$comparison"
echo "Comparison outputs: $comparison"

# Separate processes free each fitted model before the next experiment starts.
# Record returned run paths; never guess the newest directory or parse progress.
train_variant() {
    local recipe="$1"
    local label="$2"
    echo "Training $label"
    uv run --frozen python - "$recipe" "$comparison/$label-training-result.json" <<'PY'
import json
import logging
import sys
from pathlib import Path

from kelpseg.paths import check_paths, load_paths
from kelpseg.training.run import run_training
from kelpseg.training.settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
settings = load_settings(sys.argv[1])
result = run_training(settings, check_paths(load_paths(settings.paths_file)))
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, default=str) + "\n")
print(f"Completed training: {result['output']}")
PY
}
train_variant "$encoder_recipe" ade20k-encoder
train_variant "$full_recipe" ade20k-full
encoder_run=$(uv run --frozen python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$comparison/ade20k-encoder-training-result.json")
full_run=$(uv run --frozen python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$comparison/ade20k-full-training-result.json")

# All three models get fresh float32 predictions and identical report options.
make_report() {
    local run_directory="$1"
    local label="$2"
    echo "Generating $label validation predictions and report"
    uv run --frozen kelpseg predict-chips \
        --run "$run_directory" --split val --device cuda \
        --output "$comparison/$label-predictions"
    cat > "$comparison/$label-report.toml" <<EOF
predictions = "$label-predictions"
output = "$label-report"
max_examples = 12
EOF
    uv run --frozen kelpseg report --recipe "$comparison/$label-report.toml"
}
make_report "$baseline_run" baseline
make_report "$encoder_run" ade20k-encoder
make_report "$full_run" ade20k-full

# Require matching scientific support before presenting metric differences.
uv run --frozen python - "$comparison" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = ("baseline", "ade20k-encoder", "ade20k-full")
reports = [
    json.loads((root / f"{label}-report/summary.json").read_text()) for label in labels
]
window_keys = (
    "dataset_id",
    "release_id",
    "original_tiff",
    "original_row",
    "original_col",
    "height",
    "width",
)
windows = []
for label in labels:
    with (root / f"{label}-predictions/windows.csv").open() as stream:
        windows.append(
            sorted(
                tuple(row[key] for key in window_keys) for row in csv.DictReader(stream)
            )
        )
if any(candidate != windows[0] for candidate in windows[1:]):
    raise SystemExit(
        "Validation windows differ; do not compare these reports as matched results."
    )
for label, report in zip(labels[1:], reports[1:], strict=True):
    for key in ("scope", "split", "threshold", "coverage"):
        if reports[0][key] != report[key]:
            raise SystemExit(
                f"{label}: report {key} differs from baseline; inspect before comparing."
            )
    if (
        reports[0]["global"]["labeled_kelp_pixels"]
        != report["global"]["labeled_kelp_pixels"]
    ):
        raise SystemExit(f"{label}: labeled kelp support differs from baseline.")

metrics = (
    "iou",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "tn",
    "pixels",
    "area_bias_m2",
    "relative_area_bias",
)
rows = []
for metric in metrics:
    baseline, encoder, full = [report["global"][metric] for report in reports]
    rows.append(
        {
            "metric": metric,
            "baseline": baseline,
            "ade20k_encoder": encoder,
            "ade20k_full": full,
            "encoder_minus_baseline": None
            if encoder is None or baseline is None
            else encoder - baseline,
            "full_minus_baseline": None
            if full is None or baseline is None
            else full - baseline,
            "full_minus_encoder": None
            if full is None or encoder is None
            else full - encoder,
        }
    )

with (root / "comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# ADE20K initialization versus ImageNet B0",
    "",
    "Matched disjoint validation chips; float32 inference; threshold 0.5.",
    "",
    "All models were selected using validation IoU. These are validation diagnostics, not held-out test estimates.",
    "",
    "ADE20K encoder retains a fresh decoder; ADE20K full retains the pretrained decoder. Both use a fresh binary head.",
    "",
]
for label, report in zip(labels, reports, strict=True):
    lines.append(
        f"- [{label} report]({label}-report/report.html): `{report['run_id']}`, epoch {report['checkpoint_epoch']}, {report['weight_kind']} weights."
    )
lines += [
    "",
    "| Metric | Baseline | ADE20K encoder | ADE20K full | Encoder − baseline | Full − baseline | Full − encoder |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    values = [
        "—" if value is None else f"{value:.6g}" for value in list(row.values())[1:]
    ]
    lines.append(f"| {row['metric']} | {' | '.join(values)} |")
lines += [
    "",
    "Higher IoU/F1/precision/recall is better; lower FP/FN is better. Signed area bias should be closer to zero. Full − encoder compares decoder initialization while retaining the same ADE20K encoder source.",
    "",
]
(root / "comparison.md").write_text("\n".join(lines))
print(f"Comparison complete: {root / 'comparison.md'}")
PY
