import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from silhouette_pose.model import SilhouettePoseNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(path: str | Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = int(checkpoint.get("image_size", 64))

    model = SilhouettePoseNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    x = load_image(args.image, image_size).to(device)
    with torch.no_grad():
        q = model(x).squeeze(0).cpu().numpy()
    print(f"qw,qx,qy,qz = {q[0]:.8f},{q[1]:.8f},{q[2]:.8f},{q[3]:.8f}")


if __name__ == "__main__":
    main()
