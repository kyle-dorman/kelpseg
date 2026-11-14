"""Descriptive provenance for supported initialization choices; no downloads."""

from kelpseg import ade20k, dinov3, openearthmap

SUPPORTED_MODEL_WEIGHTS = (
    "openearthmap_b0",
    "ade20k_b0_encoder",
    "ade20k_b0_full",
)


def initialization_metadata(
    encoder_weights: str | None,
    model_weights: str | None,
    encoder_name: str | None = None,
    in_channels: int = 3,
) -> dict:
    """Return source/revision and transferred components for run/bundle records."""
    if encoder_weights == "dinov3":
        return dinov3.initialization_metadata(encoder_name, in_channels)
    if model_weights in SUPPORTED_MODEL_WEIGHTS:
        source = openearthmap if model_weights == "openearthmap_b0" else ade20k
        encoder_only = model_weights == "ade20k_b0_encoder"
        return {
            "source": model_weights,
            "repo_id": source.REPO_ID,
            "revision": source.REVISION,
            "filename": source.FILENAME,
            "weights": (
                "encoder only; fresh decoder and binary output layer"
                if encoder_only
                else "encoder and decoder; fresh binary output layer"
            ),
            "input_adaptation": "RGB kernels repeated cyclically and scaled by 3 / input channels",
            "implementation": "SMP Segformer; existing normalization, dropout and upsampling conventions",
        }
    return {
        "source": encoder_weights or "random",
        "weights": "encoder only" if encoder_weights else "random model",
    }
