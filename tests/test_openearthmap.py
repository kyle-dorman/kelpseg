"""Offline OpenEarthMap tensor conversion and controlled-recipe checks."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from kelpseg import openearthmap
from kelpseg.models import build_model
from kelpseg.training.settings import TrainingSettings, load_settings


@pytest.fixture
def source_weights() -> dict[str, torch.Tensor]:
    """Build an independent HF-format B0 fixture with distinguishable tensors."""
    weights = {}

    def add(name: str, shape: tuple[int, ...]) -> None:
        """Add one distinct constant tensor using the declared source shape."""
        weights[name] = torch.full(shape, (len(weights) + 1) / 1000)

    widths = (32, 64, 160, 256)
    heads = (1, 2, 5, 8)
    reductions = (8, 4, 2, 1)
    for stage, (width, head, reduction) in enumerate(
        zip(widths, heads, reductions, strict=True)
    ):
        assert width % head == 0
        patch = f"segformer.encoder.patch_embeddings.{stage}"
        inputs = 3 if stage == 0 else widths[stage - 1]
        kernel = 7 if stage == 0 else 3
        add(f"{patch}.proj.weight", (width, inputs, kernel, kernel))
        add(f"{patch}.proj.bias", (width,))
        for suffix in ("weight", "bias"):
            add(f"{patch}.layer_norm.{suffix}", (width,))
            add(f"segformer.encoder.layer_norm.{stage}.{suffix}", (width,))
        for block in range(2):
            prefix = f"segformer.encoder.block.{stage}.{block}"
            for name in ("layer_norm_1", "layer_norm_2"):
                for suffix in ("weight", "bias"):
                    add(f"{prefix}.{name}.{suffix}", (width,))
            for name in ("query", "key", "value"):
                add(f"{prefix}.attention.self.{name}.weight", (width, width))
                add(f"{prefix}.attention.self.{name}.bias", (width,))
            add(f"{prefix}.attention.output.dense.weight", (width, width))
            add(f"{prefix}.attention.output.dense.bias", (width,))
            if reduction > 1:
                add(
                    f"{prefix}.attention.self.sr.weight",
                    (width, width, reduction, reduction),
                )
                add(f"{prefix}.attention.self.sr.bias", (width,))
                for suffix in ("weight", "bias"):
                    add(f"{prefix}.attention.self.layer_norm.{suffix}", (width,))
            add(f"{prefix}.mlp.dense1.weight", (width * 4, width))
            add(f"{prefix}.mlp.dense1.bias", (width * 4,))
            add(f"{prefix}.mlp.dwconv.dwconv.weight", (width * 4, 1, 3, 3))
            add(f"{prefix}.mlp.dwconv.dwconv.bias", (width * 4,))
            add(f"{prefix}.mlp.dense2.weight", (width, width * 4))
            add(f"{prefix}.mlp.dense2.bias", (width,))
        add(f"decode_head.linear_c.{stage}.proj.weight", (256, width))
        add(f"decode_head.linear_c.{stage}.proj.bias", (256,))
    add("decode_head.linear_fuse.weight", (256, 1024, 1, 1))
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        add(f"decode_head.batch_norm.{suffix}", (256,))
    weights["decode_head.batch_norm.num_batches_tracked"] = torch.tensor(17)
    add("decode_head.classifier.weight", (9, 256, 1, 1))
    add("decode_head.classifier.bias", (9,))
    # Give RGB channels different values to expose order/scaling errors.
    stem = weights["segformer.encoder.patch_embeddings.0.proj.weight"]
    for channel in range(3):
        stem[:, channel].fill_((channel + 1) / 10)
    return weights


@pytest.fixture
def offline_source(monkeypatch, source_weights):
    """Replace downloads and file reads with a pinned in-memory fixture."""
    downloads = []

    def download(*, repo_id: str, filename: str, revision: str) -> str:
        """Record the requested immutable artifact without using the network."""
        downloads.append((repo_id, filename, revision))
        return "/offline/model.safetensors"

    def read(path: str, **kwargs) -> dict[str, torch.Tensor]:
        """Return fresh source entries so mutation cannot hide later failures."""
        assert str(path) == "/offline/model.safetensors"
        return dict(source_weights)

    monkeypatch.setattr(openearthmap, "hf_hub_download", download)
    monkeypatch.setattr(openearthmap, "load_file", read)
    return downloads


@pytest.mark.parametrize("channels", [3, 4, 8])
def test_encoder_decoder_transfer_and_fresh_binary_head(
    channels, source_weights, offline_source
) -> None:
    """Transfer both components while preserving new kelp output parameters."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = build_model(channels, encoder_name="mit_b0", encoder_weights=None)
        fresh_head = {
            key: value.clone()
            for key, value in model.segmentation_head.state_dict().items()
        }
        openearthmap.load_openearthmap_weights(model)
        actual = model.state_dict()
        source_stem = source_weights["segformer.encoder.patch_embeddings.0.proj.weight"]
        for channel in range(channels):
            torch.testing.assert_close(
                actual["encoder.patch_embed1.proj.weight"][:, channel],
                source_stem[:, channel % 3] * (3 / channels),
            )
        for stage, width in enumerate((32, 64, 160, 256)):
            for suffix in ("weight", "bias"):
                torch.testing.assert_close(
                    actual[f"decoder.mlp_stage.{3 - stage}.linear.{suffix}"],
                    source_weights[f"decode_head.linear_c.{stage}.proj.{suffix}"],
                    rtol=0,
                    atol=0,
                )
                for block in range(2):
                    combined = actual[
                        f"encoder.block{stage + 1}.{block}.attn.kv.{suffix}"
                    ]
                    for offset, name in ((0, "key"), (width, "value")):
                        torch.testing.assert_close(
                            combined[offset : offset + width],
                            source_weights[
                                f"segformer.encoder.block.{stage}.{block}.attention.self.{name}.{suffix}"
                            ],
                            rtol=0,
                            atol=0,
                        )
        torch.testing.assert_close(
            actual["decoder.fuse_stage.0.weight"],
            source_weights["decode_head.linear_fuse.weight"],
            rtol=0,
            atol=0,
        )
        assert actual["decoder.fuse_stage.1.num_batches_tracked"].item() == 17
        for key, expected in fresh_head.items():
            torch.testing.assert_close(
                model.segmentation_head.state_dict()[key], expected, rtol=0, atol=0
            )
        assert offline_source == [
            (
                "boehnen/satlens-segformer",
                "model.safetensors",
                "19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6",
            )
        ]
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("problem", ["missing", "wrong_shape", "unexpected"])
def test_reject_incompatible_source_weights(
    problem, source_weights, offline_source
) -> None:
    """Reject partial or incompatible checkpoints instead of partial loading."""
    if problem == "missing":
        del source_weights["decode_head.linear_fuse.weight"]
    elif problem == "wrong_shape":
        source_weights["decode_head.linear_fuse.weight"] = torch.zeros(1, 1, 1, 1)
    else:
        source_weights["unrelated.weight"] = torch.zeros(1)
    model = build_model(8, encoder_name="mit_b0", encoder_weights=None)
    with pytest.raises(ValueError):
        openearthmap.load_openearthmap_weights(model)


def test_builder_uses_full_model_initialization(offline_source) -> None:
    """Exercise the researcher-facing builder through its download boundary."""
    model = build_model(
        8,
        encoder_name="mit_b0",
        encoder_weights=None,
        model_weights="openearthmap_b0",
    )
    assert offline_source
    assert model.decoder.fuse_stage[1].num_batches_tracked.item() == 17


@pytest.mark.parametrize(
    "options",
    [
        {"encoder_name": "mit_b1", "encoder_weights": None},
        {"encoder_name": "mit_b0", "encoder_weights": "imagenet"},
        {"encoder_name": "mit_b0", "encoder_weights": None, "model_weights": "other"},
    ],
)
def test_reject_incompatible_initialization_options(options) -> None:
    """Reject ambiguous or unsupported choices before constructing a model."""
    arguments = {"model_weights": "openearthmap_b0", **options}
    with pytest.raises(ValueError):
        replace(TrainingSettings(), **arguments)
    with pytest.raises(ValueError):
        build_model(8, **arguments)


def test_openearthmap_recipe_matches_selected_baseline() -> None:
    """Only initialization and the run name differ from the selected B0 recipe."""
    root = Path(__file__).resolve().parents[1] / "configs/train"
    baseline = asdict(load_settings(root / "model_size/b0.toml"))
    comparison = asdict(load_settings(root / "initialization/openearthmap_b0.toml"))
    assert baseline["name"] != comparison["name"]
    assert baseline["encoder_weights"] == "imagenet"
    assert baseline["model_weights"] is None
    assert comparison["encoder_weights"] is None
    assert comparison["model_weights"] == "openearthmap_b0"
    for key in ("name", "encoder_weights", "model_weights"):
        baseline.pop(key)
        comparison.pop(key)
    assert comparison == baseline


def test_initialized_model_bundle_is_portable(
    tmp_path, monkeypatch, offline_source
) -> None:
    """Export transferred weights and reload predictions without the initializer."""
    from kelpseg.model_bundle import export_model_bundle, load_model_bundle
    from kelpseg.training.module import TrainingModule

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        settings = TrainingSettings(
            encoder_name="mit_b0", encoder_weights=None, model_weights="openearthmap_b0"
        )
        module = TrainingModule(settings)
        assert len(offline_source) == 1
        model = module.model.eval()
        image = torch.randn(1, 8, 64, 64)
        with torch.no_grad():
            expected = model(image)
        assert torch.isfinite(expected).all()
        checkpoint = tmp_path / "selected.ckpt"
        torch.save(
            {
                "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
                "epoch": 0,
                "global_step": 0,
                "averaging_state": {"n_averaged": 0},
            },
            checkpoint,
        )
        bundle = export_model_bundle(
            checkpoint, tmp_path / "bundle", settings, run_id="fixture"
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("Portable model must not fetch initialization weights")

        monkeypatch.setattr(openearthmap, "hf_hub_download", forbidden)
        loaded = load_model_bundle(bundle)
        assert loaded.metadata["initialization"]["revision"] == openearthmap.REVISION
        assert loaded.metadata["initialization"]["repo_id"] == openearthmap.REPO_ID
        with torch.no_grad():
            actual = loaded.model(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("failure", ["download", "file"])
def test_weight_access_errors_are_actionable(monkeypatch, failure) -> None:
    """Report ordinary unavailable/corrupt input files with checkpoint context."""

    def unavailable(*args, **kwargs):
        raise OSError("unavailable fixture")

    monkeypatch.setattr(
        openearthmap, "hf_hub_download", lambda **kwargs: "/missing/model.safetensors"
    )
    if failure == "download":
        monkeypatch.setattr(openearthmap, "hf_hub_download", unavailable)
    else:
        monkeypatch.setattr(openearthmap, "load_file", unavailable)
    with pytest.raises(ValueError, match="OpenEarthMap"):
        build_model(
            8,
            encoder_name="mit_b0",
            encoder_weights=None,
            model_weights="openearthmap_b0",
        )
