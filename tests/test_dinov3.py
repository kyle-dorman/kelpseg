"""False-color and eight-band DINOv3 segmentation and offline portable-model checks."""

from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from kelpseg.model_bundle import export_model_bundle, load_model_bundle
from kelpseg.models import build_model
from kelpseg.preprocessing import BAND_NAMES, MEAN, STD
from kelpseg.training.settings import TrainingSettings, load_settings

ENCODERS = ("dinov3_vits16", "dinov3_convnext_tiny")
FALSE_COLOR = ("nir", "red", "green")


@pytest.fixture(autouse=True)
def bounded_threads() -> Iterator[None]:
    """Limit CPU model fixtures and restore the caller's thread setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def false_color_settings(encoder: str, channels: int = 3) -> TrainingSettings:
    """Return an offline recipe with the approved NIR/red/green statistics."""
    names = FALSE_COLOR if channels == 3 else BAND_NAMES
    indices = [BAND_NAMES.index(name) for name in names]
    return TrainingSettings(
        encoder_name=encoder,
        encoder_weights=None,
        band_names=names,
        mean=tuple(MEAN[index] for index in indices),
        std=tuple(STD[index] for index in indices),
    )


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("channels", [3, 8])
def test_dinov3_full_resolution_and_encoder_gradients(encoder, channels) -> None:
    """Both real architectures preserve output size and train encoder/decoder."""
    model = build_model(channels, encoder_name=encoder, encoder_weights=None)
    image = torch.randn(2, channels, 64, 64)
    logits = model(image)
    assert logits.shape == (2, 1, 64, 64)
    assert logits.isfinite().all()
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.rand_like(logits)
    ).backward()
    for component in (model.encoder, model.decoder, model.segmentation_head):
        grads = [p.grad for p in component.parameters() if p.grad is not None]
        assert grads
        assert all(grad.isfinite().all() for grad in grads)
        assert any(grad.abs().sum() > 0 for grad in grads)


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("channels", [3, 8])
def test_dinov3_bundle_reconstructs_without_pretrained_access(
    encoder, channels, tmp_path, monkeypatch
) -> None:
    """Copy only bundle contents into an unrelated directory and reproduce logits."""
    import shutil

    import huggingface_hub

    settings = false_color_settings(encoder, channels)
    model = build_model(channels, encoder_name=encoder, encoder_weights=None).eval()
    image = torch.randn(1, channels, 64, 64)
    with torch.no_grad():
        expected = model(image)
    checkpoint = tmp_path / "best.ckpt"
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
        checkpoint, tmp_path / "original", settings, run_id="fixture"
    )
    copied = tmp_path / "copied"
    shutil.copytree(bundle, copied)
    shutil.rmtree(bundle)
    checkpoint.unlink()

    def forbidden(*args, **kwargs):
        """Fail if portable reconstruction requests online initialization."""
        raise AssertionError("Offline bundle must not download pretrained weights")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    import kelpseg.dinov3 as dinov3

    if hasattr(dinov3, "hf_hub_download"):
        monkeypatch.setattr(dinov3, "hf_hub_download", forbidden)
    loaded = load_model_bundle(copied)
    assert loaded.metadata["band_names"] == list(settings.band_names)
    assert loaded.metadata["encoder"] == encoder
    with torch.no_grad():
        actual = loaded.model(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_dinov3_rejects_imagenet_or_segmentation_initialization(encoder) -> None:
    """Prevent mislabeled weight sources before constructing/downloading models."""
    settings = false_color_settings(encoder)
    with pytest.raises(ValueError):
        replace(settings, encoder_weights="imagenet")
    with pytest.raises(ValueError):
        replace(settings, model_weights="ade20k_b0_full")
    with pytest.raises(ValueError):
        build_model(3, encoder_name=encoder, encoder_weights="imagenet")


@pytest.mark.parametrize("suffix,names", [("", FALSE_COLOR), ("_planet8", BAND_NAMES)])
def test_dinov3_recipes_share_false_color_and_training_policy(suffix, names) -> None:
    """Only encoder/run identity differs between the two controlled candidates."""
    root = Path(__file__).resolve().parents[1] / "configs/train/dinov3"
    recipes = [
        load_settings(root / f"{name}{suffix}.toml")
        for name in ("vits16", "convnext_tiny")
    ]
    for recipe, encoder in zip(recipes, ENCODERS, strict=True):
        assert recipe.encoder_name == encoder
        assert recipe.encoder_weights == "dinov3"
        assert recipe.band_names == names
        assert (recipe.batch_size, recipe.accumulate_grad_batches) == (12, 2)
        assert recipe.encoder_lr_scale == 0.1
        assert recipe.variant == "augmented"
        assert recipe.model_weights is None
    settings = [asdict(recipe) for recipe in recipes]
    for value in settings:
        value.pop("encoder_name")
        value.pop("name")
    assert settings[0] == settings[1]


def test_encoder_learning_rate_ratio_and_resume() -> None:
    """Warmup, cosine floor and restored optimizer retain the encoder LR ratio."""
    from kelpseg.training.module import parameter_groups
    from kelpseg.training.scheduler import LinearWarmupCosineDecayLR

    model = torch.nn.Module()
    model.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.LayerNorm(4))
    model.decoder = torch.nn.Conv2d(4, 1, 1)
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    groups = parameter_groups(model, learning_rate=0.00015, encoder_lr_scale=0.1)
    flattened = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == {id(parameter) for parameter in model.parameters()}
    for group in groups:
        assert all(
            (id(p) in encoder_ids) == (group["lr_scale"] == 0.1)
            for p in group["params"]
        )
    optimizer = torch.optim.AdamW(groups, lr=0.00015)
    scheduler = LinearWarmupCosineDecayLR(
        optimizer, total_steps=10, warmup_steps=2, min_lr=3e-6
    )
    assert all(group["lr"] == 0 for group in optimizer.param_groups)
    for step in range(1, 11):
        optimizer.step()
        scheduler.step()
        rates = {group["lr_scale"]: group["lr"] for group in optimizer.param_groups}
        assert rates[0.1] == pytest.approx(rates[1.0] * 0.1)
        if step == 2:
            assert rates[1.0] == pytest.approx(0.00015)
        if step == 5:
            saved_optimizer = optimizer.state_dict()
            saved_scheduler = scheduler.state_dict()
    assert rates[1.0] == pytest.approx(3e-6)
    restored_optimizer = torch.optim.AdamW(
        parameter_groups(model, learning_rate=0.00015, encoder_lr_scale=0.1)
    )
    restored_scheduler = LinearWarmupCosineDecayLR(restored_optimizer, 10, 2, 3e-6)
    restored_optimizer.load_state_dict(saved_optimizer)
    restored_scheduler.load_state_dict(saved_scheduler)
    for _ in range(5):
        restored_optimizer.step()
        restored_scheduler.step()
    assert restored_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())


@pytest.mark.parametrize("value", [0, -1, 2, True, float("nan"), float("inf")])
def test_reject_invalid_encoder_learning_rate_scale(value) -> None:
    """Reject scales which would freeze, reverse or invalidate optimization."""
    with pytest.raises(ValueError, match="encoder_lr_scale"):
        replace(false_color_settings(ENCODERS[0]), encoder_lr_scale=value)


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("channels", [3, 8])
def test_dinov3_dry_run_records_source_without_downloading(
    encoder, channels, tmp_path, monkeypatch
) -> None:
    """Preview false-color source metadata using manifests only, with no model IO."""
    from kelpseg import dinov3
    from kelpseg.training.run import run_training
    from tests.test_training_run import _inputs

    # The ordinary fixture intentionally has no held-out test image payload.
    settings, paths = _inputs(tmp_path)
    spectral = false_color_settings(encoder, channels)
    settings = replace(
        settings,
        encoder_name=encoder,
        encoder_weights="dinov3",
        band_names=spectral.band_names,
        mean=spectral.mean,
        std=spectral.std,
        encoder_lr_scale=0.1,
    )

    def forbidden(*args, **kwargs):
        """Fail on pretrained access during metadata-only preview."""
        raise AssertionError("Dry run must not download initialization")

    monkeypatch.setattr(dinov3, "hf_hub_download", forbidden)
    report = run_training(settings, paths, dry_run=True)
    source = report["initialization"]
    assert source["source"] == "dinov3"
    assert source["repo_id"].startswith("timm/")
    assert len(source["revision"]) == 40
    assert report["settings"]["band_names"] == spectral.band_names
    if channels == 8:
        assert (
            "3 / 8" in source["input_adaptation"] or "3/8" in source["input_adaptation"]
        )


@pytest.mark.parametrize("encoder", ENCODERS)
def test_dinov3_rejects_unsupported_four_band_input(encoder) -> None:
    """Keep the experiment scope explicit instead of accepting untested inputs."""
    with pytest.raises(ValueError, match="three or eight"):
        build_model(4, encoder_name=encoder, encoder_weights=None)
    names = ("red", "green", "blue", "nir")
    indices = [BAND_NAMES.index(name) for name in names]
    with pytest.raises(ValueError, match="three or eight"):
        replace(
            false_color_settings(encoder),
            band_names=names,
            mean=tuple(MEAN[index] for index in indices),
            std=tuple(STD[index] for index in indices),
        )


@pytest.mark.parametrize("recipe_name", ["vits16", "convnext_tiny"])
def test_eight_band_recipe_only_changes_spectral_inputs_and_identity(
    recipe_name,
) -> None:
    """Keep encoder, assignments and optimization matched across spectral variants."""
    root = Path(__file__).resolve().parents[1] / "configs/train/dinov3"
    false_color = asdict(load_settings(root / f"{recipe_name}.toml"))
    planet8 = asdict(load_settings(root / f"{recipe_name}_planet8.toml"))
    assert planet8["band_names"] == BAND_NAMES
    assert planet8["mean"] == MEAN
    assert planet8["std"] == STD
    assert planet8["name"] != false_color["name"]
    for key in ("name", "band_names", "mean", "std"):
        false_color.pop(key)
        planet8.pop(key)
    assert false_color == planet8
