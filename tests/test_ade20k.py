"""Offline checks for controlled ADE20K encoder-only and full initialization."""

from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from kelpseg import ade20k
from kelpseg.initialization import initialization_metadata
from kelpseg.model_bundle import export_model_bundle, load_model_bundle
from kelpseg.models import build_model
from kelpseg.training.settings import TrainingSettings, load_settings
from tests.test_openearthmap import source_weights as source_weights

OPTIONS = ("ade20k_b0_encoder", "ade20k_b0_full")


@pytest.fixture(autouse=True)
def bounded_cpu_threads() -> Iterator[None]:
    """Bound tensor-fixture/model checks and restore the caller's thread setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def ade_source(source_weights) -> dict[str, torch.Tensor]:
    """Reuse the independent HF B0 schema with ADE20K's 150-class output head."""
    source_weights["decode_head.classifier.weight"] = torch.full((150, 256, 1, 1), 0.5)
    source_weights["decode_head.classifier.bias"] = torch.full((150,), 0.6)
    return source_weights


@pytest.fixture
def offline_ade(monkeypatch, ade_source) -> list[tuple[str, str, str]]:
    """Serve the ADE20K fixture without network or serialized weight artifacts."""
    requests = []

    def download(*, repo_id: str, revision: str, filename: str) -> str:
        """Record the exact requested source and return an inert local path."""
        requests.append((repo_id, revision, filename))
        return "/offline/ade20k.safetensors"

    def read(path: str, **kwargs) -> dict[str, torch.Tensor]:
        """Return fixture entries while leaving the fixture dictionary intact."""
        assert str(path) == "/offline/ade20k.safetensors"
        return dict(ade_source)

    monkeypatch.setattr(ade20k, "hf_hub_download", download)
    monkeypatch.setattr(ade20k, "load_file", read)
    return requests


@pytest.mark.parametrize("channels", [3, 4, 8])
def test_modes_share_encoder_but_only_full_transfers_decoder(
    channels, ade_source, offline_ade
) -> None:
    """Compare every retained component against the same-seed random model."""
    torch.manual_seed(42)
    random_model = build_model(channels, encoder_name="mit_b0", encoder_weights=None)
    torch.manual_seed(42)
    encoder_model = build_model(
        channels,
        encoder_name="mit_b0",
        encoder_weights=None,
        model_weights="ade20k_b0_encoder",
    )
    torch.manual_seed(42)
    full_model = build_model(
        channels,
        encoder_name="mit_b0",
        encoder_weights=None,
        model_weights="ade20k_b0_full",
    )
    random = random_model.state_dict()
    encoder = encoder_model.state_dict()
    full = full_model.state_dict()
    assert random.keys() == encoder.keys() == full.keys()
    for key in random:
        if key.startswith("encoder."):
            torch.testing.assert_close(encoder[key], full[key], rtol=0, atol=0)
        else:
            torch.testing.assert_close(encoder[key], random[key], rtol=0, atol=0)
        if key.startswith("segmentation_head."):
            torch.testing.assert_close(full[key], random[key], rtol=0, atol=0)
    source_stem = ade_source["segformer.encoder.patch_embeddings.0.proj.weight"]
    for channel in range(channels):
        torch.testing.assert_close(
            encoder["encoder.patch_embed1.proj.weight"][:, channel],
            source_stem[:, channel % 3] * (3 / channels),
        )
    # Check source values as well as agreement between the two loading modes.
    torch.testing.assert_close(
        encoder["encoder.block1.0.attn.q.weight"],
        ade_source["segformer.encoder.block.0.0.attention.self.query.weight"],
        rtol=0,
        atol=0,
    )
    for stage in range(4):
        for suffix in ("weight", "bias"):
            torch.testing.assert_close(
                full[f"decoder.mlp_stage.{3 - stage}.linear.{suffix}"],
                ade_source[f"decode_head.linear_c.{stage}.proj.{suffix}"],
                rtol=0,
                atol=0,
            )
    torch.testing.assert_close(
        full["decoder.fuse_stage.0.weight"],
        ade_source["decode_head.linear_fuse.weight"],
        rtol=0,
        atol=0,
    )
    for suffix in (
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ):
        torch.testing.assert_close(
            full[f"decoder.fuse_stage.1.{suffix}"],
            ade_source[f"decode_head.batch_norm.{suffix}"],
            rtol=0,
            atol=0,
        )
    assert (
        offline_ade
        == [
            (
                "nvidia/segformer-b0-finetuned-ade-512-512",
                "489d5cd81a0b59fab9b7ea758d3548ebe99677da",
                "model.safetensors",
            )
        ]
        * 2
    )


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize(
    "changes",
    [
        {"encoder_name": "mit_b1"},
        {"encoder_weights": "imagenet"},
    ],
)
def test_reject_incompatible_ade20k_options(option, changes) -> None:
    """Reject other model sizes or mixed ImageNet/ADE20K initialization."""
    arguments = {
        "encoder_name": "mit_b0",
        "encoder_weights": None,
        "model_weights": option,
        **changes,
    }
    with pytest.raises(ValueError):
        replace(TrainingSettings(), **arguments)
    with pytest.raises(ValueError):
        build_model(8, **arguments)


@pytest.mark.parametrize("option", OPTIONS)
def test_ade20k_recipe_matches_selected_baseline(option) -> None:
    """Preserve every baseline training choice outside name and initialization."""
    root = Path(__file__).resolve().parents[1] / "configs/train"
    baseline = asdict(load_settings(root / "model_size/b0.toml"))
    candidate = asdict(load_settings(root / "initialization" / f"{option}.toml"))
    assert candidate["model_weights"] == option
    assert candidate["encoder_weights"] is None
    assert candidate["name"] != baseline["name"]
    for key in ("name", "encoder_weights", "model_weights"):
        baseline.pop(key)
        candidate.pop(key)
    assert baseline == candidate


@pytest.mark.parametrize("option", OPTIONS)
def test_ade20k_bundle_reloads_offline_with_initialization_metadata(
    option, tmp_path, monkeypatch, offline_ade
) -> None:
    """Export transferred models, disable source access and reproduce predictions."""
    settings = TrainingSettings(
        encoder_name="mit_b0", encoder_weights=None, model_weights=option
    )
    model = build_model(
        8, encoder_name="mit_b0", encoder_weights=None, model_weights=option
    ).eval()
    image = torch.randn(1, 8, 64, 64)
    with torch.no_grad():
        expected = model(image)
    assert torch.isfinite(expected).all()
    checkpoint = tmp_path / "selected.ckpt"
    torch.save(
        {
            "state_dict": {
                f"model.{key}": value for key, value in model.state_dict().items()
            },
            "epoch": 0,
            "global_step": 0,
            "averaging_state": {"n_averaged": 0},
        },
        checkpoint,
    )
    bundle = export_model_bundle(
        checkpoint, tmp_path / "model", settings, run_id="fixture"
    )

    def forbidden(*args, **kwargs):
        """Fail if offline bundle loading touches the original weight source."""
        raise AssertionError("Bundle loading must not fetch ADE20K weights")

    monkeypatch.setattr(ade20k, "hf_hub_download", forbidden)
    monkeypatch.setattr(ade20k, "load_file", forbidden)
    loaded = load_model_bundle(bundle)
    descriptor = loaded.metadata["initialization"]
    assert descriptor == initialization_metadata(None, option)
    assert descriptor["source"] == option
    assert descriptor["repo_id"] == "nvidia/segformer-b0-finetuned-ade-512-512"
    assert descriptor["revision"] == "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
    assert descriptor["weights"] == (
        "encoder only; fresh decoder and binary output layer"
        if option == "ade20k_b0_encoder"
        else "encoder and decoder; fresh binary output layer"
    )
    with torch.no_grad():
        actual = loaded.model(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(offline_ade) == 1


@pytest.mark.parametrize("include_decoder", [False, True])
def test_ade20k_rejects_other_dataset_classifier(
    include_decoder, ade_source, offline_ade
) -> None:
    """Reject an OpenEarthMap-shaped source in either ADE20K transfer mode."""
    ade_source["decode_head.classifier.weight"] = torch.zeros(9, 256, 1, 1)
    ade_source["decode_head.classifier.bias"] = torch.zeros(9)
    model = build_model(8, encoder_name="mit_b0", encoder_weights=None)
    with pytest.raises(ValueError):
        ade20k.load_ade20k_weights(model, include_decoder=include_decoder)
