"""Two DINOv3 feature encoders with a fresh, shared SegFormer decoder."""

import timm
import torch
from huggingface_hub import hf_hub_download
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base.initialization import (
    initialize_decoder,
    initialize_head,
)
from segmentation_models_pytorch.decoders.segformer.decoder import SegformerDecoder
from torch import nn

DINO_ENCODERS = {
    "dinov3_vits16": {
        "model": "vit_small_patch16_dinov3.lvd1689m",
        "revision": "3bf4720a82ec2066db88137180ff1f83a675cef0",
        "out_indices": (2, 5, 8, 11),
    },
    "dinov3_convnext_tiny": {
        "model": "convnext_tiny.dinov3_lvd1689m",
        "revision": "39c7bb32f83c11bbb13cb23058f8852a4dac88ed",
        "out_indices": (0, 1, 2, 3),
    },
}


def initialization_metadata(encoder_name: str, in_channels: int = 3) -> dict:
    """Describe the pinned web-image encoder source without downloading weights."""
    source = DINO_ENCODERS[encoder_name]
    return {
        "source": "dinov3",
        "repo_id": f"timm/{source['model']}",
        "revision": source["revision"],
        "filename": "model.safetensors",
        "weights": "encoder only; fresh SegFormer decoder and binary output layer",
        "feature_indices": list(source["out_indices"]),
        "input_adaptation": (
            "three channels unchanged; physical band order is recorded separately"
            if in_channels == 3
            else "RGB kernels repeated cyclically and scaled by 3 / 8; positional adaptation to native eight bands"
        ),
        "implementation": "timm DINOv3 features and SMP Segformer decoder (256 channels)",
    }


class DinoSegformer(nn.Module):
    """Combine four DINOv3 feature maps with a fresh full-resolution binary head."""

    def __init__(
        self, encoder_name: str, *, pretrained: bool, in_channels: int = 3
    ) -> None:
        """Build a three/eight-channel model and optionally fetch pinned encoder weights.

        ViT intermediate features all have stride 16; ConvNeXt supplies native
        strides 4/8/16/32. The decoder projects and resizes all four maps to
        stride 4 before fusion. False-color input retains the RGB-pretrained
        kernels unchanged. Eight bands repeat RGB kernels cyclically, scaled
        by 3/8; both inputs learn their new spectral meanings during fine tuning.
        """
        super().__init__()
        if type(in_channels) is not int or in_channels not in (3, 8):
            raise ValueError("DINOv3 experiments require three or eight input bands.")
        self.in_channels = in_channels
        source = DINO_ENCODERS[encoder_name]
        options = {}
        if pretrained:
            metadata = initialization_metadata(encoder_name, in_channels)
            checkpoint = hf_hub_download(
                repo_id=metadata["repo_id"],
                revision=metadata["revision"],
                filename=metadata["filename"],
            )
            options["pretrained_cfg_overlay"] = {"file": checkpoint}
        self.encoder = timm.create_model(
            source["model"],
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=source["out_indices"],
            **options,
        )
        self.decoder = SegformerDecoder(
            encoder_channels=[in_channels, 0, *self.encoder.feature_info.channels()],
            segmentation_channels=256,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=256,
            out_channels=1,
            kernel_size=1,
            upsampling=4,
            activation=None,
        )
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return full-resolution logits for NCHW inputs with the configured bands."""
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(
                f"DINOv3 expects NCHW tensors with {self.in_channels} channels."
            )
        if image.shape[2] % 32 or image.shape[3] % 32:
            raise ValueError("DINOv3 input height and width must be multiples of 32.")
        features = self.encoder(image)
        # SMP's decoder consumes the original image for its output size and a
        # zero-channel placeholder for the unused stride-2 stage.
        dummy = image.new_empty(
            image.shape[0], 0, image.shape[2] // 2, image.shape[3] // 2
        )
        decoded = self.decoder([image, dummy, *features])
        return self.segmentation_head(decoded)
