import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic_2d")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=64)
    return parser.parse_args()


def yaw_to_quaternion(angle_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(angle_rad / 2), 0.0, 0.0, math.sin(angle_rad / 2))


def rotate_point(
    x: float,
    y: float,
    angle: float,
    cx: float,
    cy: float,
) -> tuple[float, float]:
    dx = x - cx
    dy = y - cy
    ca = math.cos(angle)
    sa = math.sin(angle)
    return (cx + ca * dx - sa * dy, cy + sa * dx + ca * dy)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "silhouettes"
    image_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"

    size = args.image_size
    center = size / 2
    # An asymmetric arrow-like polygon. This is a smoke-test dataset, not a
    # replacement for CAD-rendered 3D silhouettes.
    base = [
        (center - 18, center - 8),
        (center + 6, center - 8),
        (center + 6, center - 16),
        (center + 22, center),
        (center + 6, center + 16),
        (center + 6, center + 8),
        (center - 18, center + 8),
    ]

    with labels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "qw", "qx", "qy", "qz"])
        for i in range(args.num_samples):
            angle = 2 * math.pi * i / args.num_samples
            points = [rotate_point(x, y, angle, center, center) for x, y in base]
            image = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(image)
            draw.polygon(points, fill=255)

            rel_path = f"silhouettes/sample_{i:06d}.png"
            image.save(out_dir / rel_path)
            writer.writerow([rel_path, *yaw_to_quaternion(angle)])

    print(f"wrote {args.num_samples} samples to {out_dir}")
    print(f"labels: {labels_path}")


if __name__ == "__main__":
    main()
