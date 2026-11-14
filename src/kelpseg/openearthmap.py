"""Import one pinned RGB OpenEarthMap checkpoint into the maintained SMP B0."""

import logging

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from safetensors import SafetensorError
from safetensors.torch import load_file
from torch import nn

from kelpseg.segformer_weights import apply_b0_weights, convert_b0_weights

REPO_ID = "boehnen/satlens-segformer"
REVISION = "19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6"
FILENAME = "model.safetensors"
LOGGER = logging.getLogger(__name__)


def load_openearthmap_weights(model: nn.Module) -> None:
    """Initialize an SMP B0 encoder/decoder from the pinned cached/downloaded file.

    Model must have 3, 4 or 8 inputs and a fresh binary head. Preserve that head,
    adapt RGB input kernels with the baseline's cyclic 3/channel-count rule, and
    load all remaining parameters/buffers. Weights use CPU memory. Download or
    format errors report how to select the supported checkpoint; no fit occurs.
    """
    LOGGER.info(
        "Loading OpenEarthMap B0 encoder and decoder: %s @ %s", REPO_ID, REVISION
    )
    try:
        path = hf_hub_download(repo_id=REPO_ID, revision=REVISION, filename=FILENAME)
    except (OSError, HfHubHTTPError) as error:
        raise ValueError(
            f"Cannot download/cache OpenEarthMap weights from {REPO_ID} @ {REVISION}. "
            "Check network access and the Hugging Face cache (HF_HOME), then retry."
        ) from error
    try:
        source = load_file(path, device="cpu")
    except (OSError, SafetensorError) as error:
        raise ValueError(
            f"Cannot read OpenEarthMap safetensors weights at {path}. "
            f"Use {FILENAME} from {REPO_ID} @ {REVISION}."
        ) from error
    converted = convert_b0_weights(source, source_classes=9)
    apply_b0_weights(model, converted, include_decoder=True)
    LOGGER.info("Loaded OpenEarthMap encoder/decoder; retained fresh binary head")
