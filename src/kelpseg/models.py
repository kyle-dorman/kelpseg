"""SegFormer models with explicit encoder size and shared binary output."""

import segmentation_models_pytorch as smp
from torch import nn

from kelpseg.ade20k import load_ade20k_weights
from kelpseg.dinov3 import DINO_ENCODERS, DinoSegformer
from kelpseg.initialization import SUPPORTED_MODEL_WEIGHTS
from kelpseg.openearthmap import load_openearthmap_weights

SUPPORTED_ENCODERS = ("mit_b0", "mit_b1", "mit_b2", "mit_b3", *DINO_ENCODERS)


def build_model(
    in_channels: int,
    *,
    encoder_weights: str | None = "imagenet",
    encoder_name: str = "mit_b3",
    model_weights: str | None = None,
) -> nn.Module:
    """Build the selected SegFormer encoder with one full-resolution foreground logit.

    Set ``model_weights="openearthmap_b0"`` with B0 and no encoder weights
    to initialize the encoder and decoder from the pinned RGB checkpoint.
    ADE20K options are "ade20k_b0_encoder" (fresh decoder) and "ade20k_b0_full"
    (pretrained decoder); both retain a fresh binary output layer.
    DINOv3 encoders use encoder_weights="dinov3" with three or eight bands
    and a fresh SegFormer decoder. Leave both weight options None for offline
    construction, including portable bundle loading. ImageNet initialization
    uses SMP's first-convolution adaptation: RGB weights repeat cyclically and
    are scaled by 3 / in_channels for four/eight inputs. Three inputs retain
    the original RGB weights. This adaptation is positional, not a learned
    physical-band mapping; the loader owns the recorded physical band order.
    """
    if type(in_channels) is not int or in_channels not in (3, 4, 8):
        raise ValueError("SegFormer requires 3, 4, or 8 input bands.")
    if encoder_name not in SUPPORTED_ENCODERS:
        raise ValueError(f"encoder_name must be one of {SUPPORTED_ENCODERS}.")
    validate_model_weights(encoder_name, encoder_weights, model_weights)
    if encoder_name in DINO_ENCODERS:
        return DinoSegformer(
            encoder_name,
            pretrained=encoder_weights == "dinov3",
            in_channels=in_channels,
        )
    model = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
        activation=None,
    )
    if model_weights == "openearthmap_b0":
        load_openearthmap_weights(model)
    elif model_weights in ("ade20k_b0_encoder", "ade20k_b0_full"):
        load_ade20k_weights(model, include_decoder=model_weights == "ade20k_b0_full")
    return model


def validate_model_weights(
    encoder_name: str, encoder_weights: str | None, model_weights: str | None
) -> None:
    """Reject unsupported or ambiguous full-model initialization before downloading."""
    allowed_weights = (
        (None, "dinov3") if encoder_name in DINO_ENCODERS else (None, "imagenet")
    )
    if encoder_weights not in allowed_weights:
        raise ValueError(
            f"encoder_weights for {encoder_name} must be one of {allowed_weights}."
        )
    if model_weights is not None and model_weights not in SUPPORTED_MODEL_WEIGHTS:
        raise ValueError(
            f"model_weights must be one of {SUPPORTED_MODEL_WEIGHTS} or omitted."
        )
    if model_weights is not None and (
        encoder_name != "mit_b0" or encoder_weights is not None
    ):
        raise ValueError(
            "Segmentation model_weights requires encoder_name='mit_b0' "
            "and encoder_weights='none' in recipes (None in Python)."
        )
