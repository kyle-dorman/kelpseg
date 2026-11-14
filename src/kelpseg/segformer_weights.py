"""Translate the two supported Hugging Face B0 segmentation checkpoints to SMP."""

import math

import torch
from torch import Tensor, nn


def convert_b0_weights(
    source: dict[str, Tensor], *, source_classes: int
) -> dict[str, Tensor]:
    """Map the pinned HF B0 tensors into SMP names, excluding its source-class head.

    Source is the safetensors state dictionary. Return encoder/decoder tensors;
    combine attention key/value projections and reverse decoder stage indices.
    Missing or unexpected source entries raise ValueError rather than silently
    leaving random encoder/decoder parameters.
    """
    converted = {}
    used = set()

    def copy(destination: str, origin: str) -> None:
        """Copy one source tensor and record its consumption; missing keys fail."""
        converted[destination] = source[origin]
        used.add(origin)

    try:
        for stage in range(4):
            hf = f"segformer.encoder.patch_embeddings.{stage}"
            smp = f"encoder.patch_embed{stage + 1}"
            for suffix in ("weight", "bias"):
                copy(f"{smp}.proj.{suffix}", f"{hf}.proj.{suffix}")
                copy(f"{smp}.norm.{suffix}", f"{hf}.layer_norm.{suffix}")
                copy(
                    f"encoder.norm{stage + 1}.{suffix}",
                    f"segformer.encoder.layer_norm.{stage}.{suffix}",
                )
                copy(
                    f"decoder.mlp_stage.{3 - stage}.linear.{suffix}",
                    f"decode_head.linear_c.{stage}.proj.{suffix}",
                )
            for block in range(2):
                hf = f"segformer.encoder.block.{stage}.{block}"
                smp = f"encoder.block{stage + 1}.{block}"
                layers = {
                    "norm1": "layer_norm_1",
                    "norm2": "layer_norm_2",
                    "attn.q": "attention.self.query",
                    "attn.proj": "attention.output.dense",
                    "mlp.fc1": "mlp.dense1",
                    "mlp.fc2": "mlp.dense2",
                    "mlp.dwconv.dwconv": "mlp.dwconv.dwconv",
                }
                if stage < 3:
                    layers.update(
                        {
                            "attn.sr": "attention.self.sr",
                            "attn.norm": "attention.self.layer_norm",
                        }
                    )
                for destination, origin in layers.items():
                    for suffix in ("weight", "bias"):
                        copy(f"{smp}.{destination}.{suffix}", f"{hf}.{origin}.{suffix}")
                for suffix in ("weight", "bias"):
                    key = f"{hf}.attention.self.key.{suffix}"
                    value = f"{hf}.attention.self.value.{suffix}"
                    converted[f"{smp}.attn.kv.{suffix}"] = torch.cat(
                        [source[key], source[value]], dim=0
                    )
                    used.update((key, value))
        copy("decoder.fuse_stage.0.weight", "decode_head.linear_fuse.weight")
        for suffix in (
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ):
            copy(f"decoder.fuse_stage.1.{suffix}", f"decode_head.batch_norm.{suffix}")
    except (KeyError, RuntimeError) as error:
        raise ValueError(
            f"SegFormer B0 checkpoint has missing or incompatible tensors: {error}"
        ) from error
    ignored = {"decode_head.classifier.weight", "decode_head.classifier.bias"}
    if set(source) - used != ignored:
        raise ValueError(
            "SegFormer B0 checkpoint has unexpected or missing output tensors."
        )
    if source["decode_head.classifier.weight"].shape != (
        source_classes,
        256,
        1,
        1,
    ) or source["decode_head.classifier.bias"].shape != (source_classes,):
        raise ValueError(
            f"SegFormer B0 checkpoint must have its original {source_classes}-class output layer."
        )
    return converted


def apply_b0_weights(
    model: nn.Module, converted: dict[str, Tensor], *, include_decoder: bool
) -> None:
    """Load converted weights into B0, adapting RGB kernels and retaining its head.

    Validate complete encoder/decoder shapes before mutation. For encoder-only
    transfer preserve every freshly initialized decoder parameter and buffer.
    Three inputs retain RGB kernels; four/eight repeat them and scale by 3/N.
    No network access or training occurs here.
    """
    target = model.state_dict()
    head = {"segmentation_head.0.weight", "segmentation_head.0.bias"}
    if set(target) - head != set(converted):
        raise ValueError(
            "SegFormer B0 initialization requires the standard SMP mit_b0 encoder and decoder."
        )
    stem = "encoder.patch_embed1.proj.weight"
    channels = target[stem].shape[1]
    if channels not in (3, 4, 8) or converted[stem].shape != (32, 3, 7, 7):
        raise ValueError(
            "SegFormer B0 initialization requires RGB source weights and 3, 4 or 8 model inputs."
        )
    if channels != 3:
        converted[stem] = converted[stem].repeat(1, math.ceil(channels / 3), 1, 1)[
            :, :channels
        ] * (3 / channels)
    for name, value in converted.items():
        if value.shape != target[name].shape:
            raise ValueError(
                f"SegFormer B0 tensor {name} has shape {tuple(value.shape)}; expected {tuple(target[name].shape)} for SMP B0."
            )
    if include_decoder:
        target.update(converted)
    else:
        target.update(
            {
                name: value
                for name, value in converted.items()
                if name.startswith("encoder.")
            }
        )
    model.load_state_dict(target, strict=True)
