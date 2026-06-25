# Silhouette Pose Estimation Prototype

This is a minimal PyTorch implementation of the second-stage idea from
SilhoNet:

```text
binary silhouette image -> CNN -> L2-normalized quaternion
```

It is intended as a small starting point for experiments where CAD-rendered
silhouettes are available as training data.

## Dataset Format

Prepare a CSV file with one row per silhouette image:

```csv
image,qw,qx,qy,qz
silhouettes/sample_000001.png,0.98,0.01,0.02,0.19
```

The `image` path is relative to `--data-root`. Quaternions are ordered as
`qw,qx,qy,qz`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke-Test Dataset

This script creates simple 2D arrow silhouettes with yaw-only quaternion labels.
It is just for checking the training loop.

```bash
python make_synthetic_2d_dataset.py --output-dir data/synthetic_2d --num-samples 1000
python train.py \
  --data-root data/synthetic_2d \
  --labels-csv data/synthetic_2d/labels.csv \
  --epochs 20
```

## Train On CAD-Rendered Silhouettes

After rendering silhouettes from the 3D model, train with:

```bash
python train.py \
  --data-root path/to/dataset \
  --labels-csv path/to/dataset/labels.csv \
  --epochs 100 \
  --batch-size 64
```

## Inference

```bash
python infer.py \
  --checkpoint checkpoints/silhouette_pose.pt \
  --image path/to/silhouette.png
```

## Notes

- The model input is a single-channel binary image, resized to 64x64 by default.
- The output is a unit quaternion.
- The loss treats `q` and `-q` as the same orientation.
- This prototype only estimates rotation. Translation and scale should be added
  separately, for example from apparent silhouette size or a dedicated head.
