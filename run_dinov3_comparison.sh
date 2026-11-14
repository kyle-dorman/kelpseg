#!/usr/bin/env bash
# Train ViT-S and ConvNeXt-Tiny with false color and eight bands; report against both B0 references.
# Usage: bash run_dinov3_comparison.sh [--check] [ABSOLUTE_OUTPUT_DIRECTORY]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
check_only=false
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash run_dinov3_comparison.sh [--check] [ABSOLUTE_OUTPUT_DIRECTORY]"
    echo "BASELINE_RUN may relocate the documented eight-band epoch-16 B0 run."
    echo "FALSE_COLOR_RUN may relocate the documented false-color epoch-46 B0 run."
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
vit_recipe="configs/train/dinov3/vits16.toml"
convnext_recipe="configs/train/dinov3/convnext_tiny.toml"
vit8_recipe="configs/train/dinov3/vits16_planet8.toml"
convnext8_recipe="configs/train/dinov3/convnext_tiny_planet8.toml"
baseline_id="planet8-cabc-size-b0-bparams-70ep-20261106T085748896218Z"
false_color_id="planet3-cabc-b0-spectral-037b-70ep-20261110T195653341372Z"
data_root=$(uv run --frozen python -c 'from kelpseg.paths import load_paths; print(load_paths("paths.local.toml").data_root)')
baseline_run="${BASELINE_RUN:-$data_root/model-size-runs/$baseline_id}"
false_color_run="${FALSE_COLOR_RUN:-$data_root/runs/$false_color_id}"
comparison="${1:-$data_root/comparisons/dinov3-spectral-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "$comparison" != /* || -e "$comparison" ]]; then
    echo "Choose a fresh absolute output directory: $comparison" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1

# Read-only preflight: compare actual saved baseline settings and preview scope.
uv run --frozen python - "$vit_recipe" "$convnext_recipe" "$vit8_recipe" "$convnext8_recipe" "$baseline_run" "$baseline_id" "$false_color_run" "$false_color_id" <<'PY'
import json
import sys
from pathlib import Path

import torch

from kelpseg.paths import check_paths, load_paths
from kelpseg.preprocessing import BAND_NAMES
from kelpseg.training.run import run_training
from kelpseg.training.settings import load_settings

vit_recipe, convnext_recipe, vit8_recipe, convnext8_recipe, baseline_path, baseline_id, false_color_path, false_color_id = sys.argv[1:]
references = {}
for label, path, run_id, epoch, bands, override in (
    ("planet8", baseline_path, baseline_id, 15, BAND_NAMES, "BASELINE_RUN"),
    ("false_color", false_color_path, false_color_id, 45, ("nir", "red", "green"), "FALSE_COLOR_RUN"),
):
    root = Path(path)
    for relative in ("run.json", "model/model.json", "model/weights.pt"):
        if not (root / relative).is_file():
            raise SystemExit(f"Missing reference file: {root / relative}. Set {override} to the complete selected run.")
    saved = json.loads((root / "run.json").read_text())
    model = json.loads((root / "model/model.json").read_text())
    if (
        saved["status"] != "complete"
        or saved["run_id"] != run_id
        or model["run_id"] != run_id
        or model["checkpoint_epoch"] != epoch
        or model["weight_kind"] != "ema"
        or model["threshold"] != 0.5
        or tuple(model["band_names"]) != bands
        or model["encoder"] != "mit_b0"
    ):
        raise SystemExit(f"Expected the completed {label} B0 reference, epoch {epoch + 1}, EMA, threshold 0.5, matching bands.")
    references[label] = saved
previews = []
candidate_settings = []
for recipe, expected_encoder, spectral in (
    (vit_recipe, "dinov3_vits16", "false_color"),
    (convnext_recipe, "dinov3_convnext_tiny", "false_color"),
    (vit8_recipe, "dinov3_vits16", "planet8"),
    (convnext8_recipe, "dinov3_convnext_tiny", "planet8"),
):
    settings = load_settings(recipe)
    expected_bands = BAND_NAMES if spectral == "planet8" else ("nir", "red", "green")
    if (
        settings.encoder_name != expected_encoder
        or settings.encoder_weights != "dinov3"
        or settings.model_weights is not None
        or settings.encoder_lr_scale != 0.1
        or settings.band_names != expected_bands
    ):
        raise SystemExit(f"Expected {spectral} {expected_encoder} with DINOv3 initialization and encoder LR scale 0.1 in {recipe}.")
    if (settings.batch_size, settings.accumulate_grad_batches) != (12, 2):
        raise SystemExit(f"Expected batch 12 and accumulation 2 (effective 24) in {recipe}.")
    preview = run_training(settings, check_paths(load_paths(settings.paths_file)), dry_run=True)
    current = json.loads(json.dumps(preview["settings"]))
    reference = dict(references[spectral]["settings"])
    candidate = dict(current)
    for key in ("name", "encoder_name", "band_names", "mean", "std"):
        candidate.pop(key)
    candidate_settings.append(candidate)
    for key in (
        "name", "paths_file", "encoder_name", "encoder_weights", "model_weights",
        "encoder_lr_scale",
    ):
        current.pop(key, None)
        reference.pop(key, None)
    if current != reference:
        changed = sorted(key for key in current.keys() | reference.keys() if current.get(key) != reference.get(key))
        raise SystemExit(f"{recipe} differs from its same-band B0 reference beyond declared encoder and encoder-LR changes: {changed}")
    for label, saved in references.items():
        for key in ("chips", "assignments"):
            if Path(preview[key]).resolve() != Path(saved[key]).resolve():
                raise SystemExit(f"{label} reference and candidate must use matching {key} paths.")
    previews.append({"recipe": recipe, "initialization": preview["initialization"], "counts": preview["counts"], "budget": preview["budget"]})
if any(candidate != candidate_settings[0] for candidate in candidate_settings[1:]):
    raise SystemExit("The four DINOv3 recipes must match except for encoder, run name and band-named inputs/statistics.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; run this GPU experiment on the configured GPU host.")
print(json.dumps({"references": {"planet8": baseline_path, "false_color": false_color_path}, "experiments": previews}, indent=2))
print("Four fits: ViT-S and ConvNeXt-Tiny, each with NIR/red/green and native eight bands.")
print("Both encoders and fresh decoders are trainable; six matched validation reports follow.")
print("Same-band B0 comparisons still differ in capacity, initialization and encoder LR.")
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
train_variant "$vit_recipe" dinov3-vits16
train_variant "$convnext_recipe" dinov3-convnext-tiny
train_variant "$vit8_recipe" dinov3-vits16-planet8
train_variant "$convnext8_recipe" dinov3-convnext-tiny-planet8
vit_run=$(uv run --frozen python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$comparison/dinov3-vits16-training-result.json")
convnext_run=$(uv run --frozen python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$comparison/dinov3-convnext-tiny-training-result.json")
vit8_run=$(uv run --frozen python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$comparison/dinov3-vits16-planet8-training-result.json")
convnext8_run=$(uv run --frozen python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$comparison/dinov3-convnext-tiny-planet8-training-result.json")

# All six models get fresh float32 predictions and identical report options.
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
make_report "$false_color_run" baseline-false-color
make_report "$vit_run" dinov3-vits16
make_report "$convnext_run" dinov3-convnext-tiny
make_report "$vit8_run" dinov3-vits16-planet8
make_report "$convnext8_run" dinov3-convnext-tiny-planet8

# Require matching scientific support before presenting metric differences.
uv run --frozen python - "$comparison" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = ("baseline", "baseline-false-color", "dinov3-vits16", "dinov3-convnext-tiny", "dinov3-vits16-planet8", "dinov3-convnext-tiny-planet8")
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
# Columns retain all six arms; differences use the corresponding spectral reference.
differences = (
    ("vits16_false_color_minus_b0_false_color", 2, 1),
    ("convnext_false_color_minus_b0_false_color", 3, 1),
    ("vits16_planet8_minus_b0_planet8", 4, 0),
    ("convnext_planet8_minus_b0_planet8", 5, 0),
    ("vits16_planet8_minus_false_color", 4, 2),
    ("convnext_planet8_minus_false_color", 5, 3),
    ("convnext_minus_vits16_false_color", 3, 2),
    ("convnext_minus_vits16_planet8", 5, 4),
)
rows = []
for metric in metrics:
    values = [report["global"][metric] for report in reports]
    row = {"metric": metric, **dict(zip(labels, values, strict=True))}
    for name, candidate, reference in differences:
        left, right = values[candidate], values[reference]
        row[name] = None if left is None or right is None else left - right
    rows.append(row)

with (root / "comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# DINOv3 false-color and eight-band comparison",
    "",
    "Matched disjoint validation chips; float32 inference; threshold 0.5.",
    "",
    "All models were selected using validation IoU. These are validation diagnostics, not held-out test estimates.",
    "",
    "False color is NIR/red/green. Eight-band inputs use native Planet order. DINOv3 encoders are fine-tuned with fresh SegFormer decoders and binary heads; eight-band RGB kernels repeat cyclically and scale by 3/8. B0 references match the respective bands. Same-band comparisons still differ in encoder capacity, initialization and encoder learning rate.",
    "",
]
for label, report in zip(labels, reports, strict=True):
    lines.append(f"- [{label} report]({label}-report/report.html): `{report['run_id']}`, epoch {report['checkpoint_epoch']}, {report['weight_kind']} weights.")
lines += [
    "",
    "| Metric | B0 eight-band | B0 false color | ViT-S false color | ConvNeXt false color | ViT-S eight-band | ConvNeXt eight-band |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    values = ["—" if row[label] is None else f"{row[label]:.6g}" for label in labels]
    lines.append(f"| {row['metric']} | {' | '.join(values)} |")
lines += ["", "## Differences", ""]
for name, _, _ in differences:
    lines += [f"### {name.replace('_', ' ')}", "", "| Metric | Difference |", "|---|---:|"]
    for row in rows:
        value = "—" if row[name] is None else f"{row[name]:.6g}"
        lines.append(f"| {row['metric']} | {value} |")
    lines.append("")
lines += [
    "Higher IoU/F1/precision/recall is better; lower FP/FN is better. Signed area bias should be closer to zero. Within-encoder eight-band minus false-color differences compare input bands and their initialization adaptation with matched training settings. ConvNeXt minus ViT-S compares architectures with matching bands and training settings; their capacities differ.",
    "",
]
(root / "comparison.md").write_text("\n".join(lines))
print(f"Comparison complete: {root / 'comparison.md'}")
PY
