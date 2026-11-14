"""Load ADE20K B0 encoder-only or encoder/decoder initialization into SMP."""

import logging

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from safetensors import SafetensorError
from safetensors.torch import load_file
from torch import nn

from kelpseg.segformer_weights import apply_b0_weights, convert_b0_weights

REPO_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
FILENAME = "model.safetensors"
LOGGER = logging.getLogger(__name__)


def load_ade20k_weights(model: nn.Module, *, include_decoder: bool) -> None:
    """Initialize B0 from the pinned ADE20K checkpoint, always keeping a fresh head.

    Download/cache CPU weights, validate and convert the full source checkpoint,
    then transfer its encoder and optionally decoder. Encoder-only preserves all
    fresh decoder parameters/buffers. The original 150-class classifier is never
    loaded. RGB-to-4/8-channel adaptation matches the ImageNet baseline policy.
    No fitting occurs; report download or format mistakes with source context.
    """
    components = "encoder and decoder" if include_decoder else "encoder only"
    LOGGER.info("Loading ADE20K B0 %s: %s @ %s", components, REPO_ID, REVISION)
    try:
        path = hf_hub_download(repo_id=REPO_ID, revision=REVISION, filename=FILENAME)
    except (OSError, HfHubHTTPError) as error:
        raise ValueError(
            f"Cannot download/cache ADE20K weights from {REPO_ID} @ {REVISION}. "
            "Check network access and the Hugging Face cache (HF_HOME), then retry."
        ) from error
    try:
        source = load_file(path, device="cpu")
    except (OSError, SafetensorError) as error:
        raise ValueError(
            f"Cannot read ADE20K safetensors weights at {path}. "
            f"Use {FILENAME} from {REPO_ID} @ {REVISION}."
        ) from error
    converted = convert_b0_weights(source, source_classes=150)
    apply_b0_weights(model, converted, include_decoder=include_decoder)
    LOGGER.info("Loaded ADE20K %s; retained fresh binary head", components)
