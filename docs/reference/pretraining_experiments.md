# Pretraining and encoder experiments: results and retirement

The ImageNet-initialized eight-band SegFormer B0 remains the selected model.
ADE20K encoder initialization essentially tied it; transferring the ADE20K
decoder reduced agreement slightly. DINOv3 ConvNeXt-Tiny with false color came
close to the false-color B0 reference, but neither DINOv3 encoder improved pooled
validation IoU. OpenEarthMap has an incomplete run record and no finished report.

These experiments were reviewed and retired at the user's request. Their code,
recipes and launchers were committed before removal. The preservation commit is `c5553dfa96f4f8c1017ed2e06fcb83eeb3152628`;
recovery instructions are recorded in [Task 038G](../../tasks/038g_retire_pretraining_experiments.md).
External runs, checkpoints and reports remain outside Git and were not deleted.
The maintained toolkit retains ordinary ImageNet/random MiT B0–B3 support;
DINOv3 models require the archived implementation for reconstruction.

## Questions and experimental design

The comparisons tested whether a segmentation-pretrained B0 encoder helps kelp
segmentation, whether its pretrained decoder helps beyond that encoder, and
whether larger DINOv3 encoders improve agreement with the supplied kelp labels.

All fits used the existing CA/BC canonical chips, with-BC (`augmented`) source
assignments, kelp-positive training eligibility, representative validation,
seed 42, 512-pixel inputs, 0.7 unsmoothed BCE plus 0.3 Dice, and EMA selection by
pooled validation IoU. The budget was 70 epochs, with patience 20 and five warmup
epochs. Physical batch 12 with accumulation 2 gave effective batch 24 and 355
optimizer updates per epoch. Report threshold remained 0.5. Test data were not
used for checkpoint or recipe selection.

- **B0 references:** ImageNet MiT encoder, fresh SegFormer decoder and binary head.
- **OpenEarthMap B0:** RGB segmentation-trained encoder and decoder transferred
  from `boehnen/satlens-segformer`, revision
  `19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6`; fresh binary head.
- **ADE20K B0:** the same encoder source in both arms, with either a fresh or
  transferred decoder. Source `nvidia/segformer-b0-finetuned-ade-512-512`, revision
  `489d5cd81a0b59fab9b7ea758d3548ebe99677da`; fresh binary head in both.
- **DINOv3:** ViT-S/16 or ConvNeXt-Tiny web-pretrained encoders, fresh SegFormer
  decoder/head, all weights fine-tuned. ViT used blocks 3/6/9/12 at stride 16;
  ConvNeXt supplied native strides 4/8/16/32. Encoder peak LR was 0.000015,
  one-tenth of the decoder's 0.00015; the LR floor was scaled proportionally.
  MiT reference encoders used the same LR as their decoder.

False color means NIR/red/green, retaining the three RGB-pretrained kernels in
that positional order. Eight-band inputs used native Planet band order and RGB
kernel repetition scaled by 3/8. All recipes retained approved Planet raw-unit
means/stds and the cap of 5000; neither adaptation constitutes multispectral
pretraining. DINOv3 sources were `timm/vit_small_patch16_dinov3.lvd1689m` at
`3bf4720a82ec2066db88137180ff1f83a675cef0` and
`timm/convnext_tiny.dinov3_lvd1689m` at
`39c7bb32f83c11bbb13cb23058f8852a4dac88ed`.

## Completed comparisons

The following scores come from completed float32, disjoint-chip reports on the
same 933 validation chips, 53 source TIFFs and 215,524,030 scored pixels. Epochs
are one-based selected EMA checkpoints. These are not the earlier full-TIFF
spectral study scores, which covered a different footprint.

| Model / initialization | Bands | Selected epoch | IoU % | Precision % | Recall % | IoU difference vs same-band B0, points |
|---|---|---:|---:|---:|---:|---:|
| B0 ImageNet reference | Eight | 16 | 68.553 | 82.914 | 79.831 | — |
| B0 ImageNet reference | False color | 46 | 67.737 | 81.357 | 80.183 | — |
| B0 ADE20K encoder, fresh decoder | Eight | 50 | 68.551 | 82.908 | 79.834 | −0.002 |
| B0 ADE20K encoder and decoder | Eight | 20 | 68.224 | 82.456 | 79.809 | −0.329 |
| DINOv3 ViT-S | False color | 59 | 63.347 | 77.198 | 77.928 | −4.391 |
| DINOv3 ConvNeXt-Tiny | False color | 30 | 67.587 | 80.524 | 80.794 | −0.150 |
| DINOv3 ViT-S | Eight | 67 | 63.714 | 77.442 | 78.234 | −4.839 |
| DINOv3 ConvNeXt-Tiny | Eight | 21 | 66.459 | 79.233 | 80.476 | −2.095 |

**ADE20K:** encoder-only was effectively tied with the reference but selected a
later epoch. The transferred decoder added 13,455 false positives and 642 false
negatives relative to the fresh decoder, reducing IoU by 0.328 points. This
supports keeping a fresh decoder for this task; it does not imply that a decoder
is unnecessary or that pretrained decoders universally hurt. Encoder-only ran
70 epochs (24,850 updates); full transfer ran 40 (14,200).

**DINOv3:** false-color ConvNeXt recovered 15,729 additional kelp pixels relative
to false-color B0 but introduced 30,057 extra false positives. Eight bands gained
ViT 0.368 IoU points while reducing ConvNeXt by 1.128 points. Regional effects
varied: eight-band ConvNeXt improved some regions but reduced region 004 IoU
from B0's 60.91% to 50.97%.

ViT's boundary F1 at two-pixel tolerance was 43.64%/45.10% for false color/eight
bands, versus 61.46%/63.50% for their B0 references. Recall within labeled patches
of 11–100 pixels was 23.90%/26.46%, versus B0's 43.91%/45.93%. This is consistent
with insufficient spatial detail in the coarse ViT features, but does not isolate
that mechanism from training and initialization choices.

## Capacity, fitting and interpretation

ConvNeXt-Tiny has about 27.8 million encoder parameters and ViT-S about 22 million,
compared with roughly 3.3 million for MiT B0. Capacity and encoder learning rate
therefore differ alongside pretraining; these are not isolated weight-source tests.

ConvNeXt showed increasing training fit without sustained validation gains:

- False color: training IoU rose from 77.55% at epoch 30 to 79.47% at epoch 46,
  while training-time validation IoU fell from 67.60% to 67.44%. It stopped at
  epoch 50 (17,750 updates), selecting epoch 30.
- Eight bands: training IoU reached 80.84% at epoch 30 while validation was
  62.87%, below its 67.10% peak at epoch 21. It stopped at epoch 41 (14,555 updates).
- Both ViTs ran all 70 epochs (24,850 updates), with only small late validation
  gains. False-color validation was 63.33% at both epochs 59 and 70; eight-band
  validation moved from 63.53% at epoch 46 to 63.69% at epoch 70.

The ConvNeXt curves are evidence of overfitting under this fine-tuning recipe.
They do not establish that capacity alone caused it or that frozen/partially
frozen encoders could not generalize better. No such follow-up was run. A smaller
net area bias also does not prove better mapping: false positives and missed
kelp can cancel when areas are summed.

Eight-band ConvNeXt's selected checkpoint scored 67.100% during bf16 mixed-
precision training validation but 66.459% in float32 reporting. All 194 exported
model tensors exactly matched the selected checkpoint. The discrepancy remains
unexplained; neither score beats eight-band B0. Other reported conclusions use
the common float32 report protocol rather than mixing the two measurements.

All comparisons use one seed and validation-selected models. Small gaps such as
false-color ConvNeXt versus B0 do not establish a repeatable superiority claim.
Results concern agreement with supplied labels, not independent ecological truth
or demonstrated geographic/temporal generalization.

## OpenEarthMap: incomplete evidence

The first launch has no logged validation epoch. The second retains 31 completed
validation epochs and `best.ckpt`/`last.ckpt`, but its `run.json` still says
`running`; no exported bundle or completed comparison report is present. That
status is an unfinished record, not confirmation of a currently running process.
Its best observed training-time validation IoU was **67.566% at epoch 29**,
with precision 81.991% and recall 79.341%. Do not present this partial mixed-
precision result as a completed float32 comparison or a final converged outcome.

## Run identities and external evidence

All W&B links use `kdorman90-ucla/kelpseg`. Exact IDs also name local run folders:

| Arm | W&B run |
|---|---|
| Eight-band B0 reference | [planet8-cabc-size-b0-bparams-70ep-20261106T085748896218Z](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet8-cabc-size-b0-bparams-70ep-20261106T085748896218Z) |
| False-color B0 reference | [planet3-cabc-b0-spectral-037b-70ep-20261110T195653341372Z](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet3-cabc-b0-spectral-037b-70ep-20261110T195653341372Z) |
| OpenEarthMap first launch, no validation | [planet8-cabc-b0-openearthmap-70ep-20261113T211122869046Z](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet8-cabc-b0-openearthmap-70ep-20261113T211122869046Z) |
| OpenEarthMap partial run | [planet8-cabc-b0-openearthmap-70ep-20261113T211401576918Z](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet8-cabc-b0-openearthmap-70ep-20261113T211401576918Z) |
| ADE20K encoder only | [planet8-cabc-b0-ade20k-encoder-70ep-20261113T222839108961Z](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet8-cabc-b0-ade20k-encoder-70ep-20261113T222839108961Z) |
| ADE20K encoder and decoder | [planet8-cabc-b0-ade20k-full-70ep-20261114T011254922555Z](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet8-cabc-b0-ade20k-full-70ep-20261114T011254922555Z) |
| ViT-S false color | [planet3-cabc-dinov3-vits16-70ep-20261114T040242179077Z](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet3-cabc-dinov3-vits16-70ep-20261114T040242179077Z) |
| ConvNeXt false color | [planet3-cabc-dinov3-convnext-tiny-70ep-20261114T063344208549Z](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet3-cabc-dinov3-convnext-tiny-70ep-20261114T063344208549Z) |
| ViT-S eight-band | [planet8-cabc-dinov3-vits16-70ep-20261114T081636665051Z](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet8-cabc-dinov3-vits16-70ep-20261114T081636665051Z) |
| ConvNeXt eight-band | [planet8-cabc-dinov3-convnext-tiny-70ep-20261114T110528729340Z](https://wandb.ai/kdorman90-ucla/kelpseg/runs/planet8-cabc-dinov3-convnext-tiny-70ep-20261114T110528729340Z) |

The earlier batch-3 ViT launch `planet3-cabc-dinov3-vits16-70ep-20261114T035656973373Z`
is excluded from the completed four-run comparison. Final DINOv3 runs used batch
12 × accumulation 2 after bounded RTX 3090 memory probes passed.

Local evidence roots (host paths, not a claim that remote backups were verified):

- `/home/sky/data/model-size-runs/<eight-band-reference-id>/` holds the B0 reference.
- `/home/sky/data/runs/<run-id>/` holds the other run records, metrics and checkpoints.
- `/home/sky/data/comparisons/ade20k-b0-20261113T222827Z/` contains the ADE20K
  comparison CSV/Markdown and three HTML reports.
- `/home/sky/data/comparisons/dinov3-spectral-20261114T040229Z/` contains the final
  comparison CSV/Markdown and six HTML reports, including both B0 references.
- `/home/sky/data/comparisons/openearthmap-b0-20261113T211111Z/` and
  `/home/sky/data/comparisons/openearthmap-b0-20261113T211350Z/` are incomplete
  launcher output folders, not completed comparisons.
- `/home/sky/data/analyses/dinov3-batch12-memory-20261114/` retains bounded GPU
  probe code/results. It contains no production-trained models.
