"""
Convert Euler angle labels (from MATLAB rendering) to quaternion CSV
for use with silhouette_pose train.py.

Rotation convention (matches renderBoxImage.m):
    R = Rz(roll) * Ry(yaw) * Rx(pitch)

Usage:
    python make_labels_csv.py \
        --input  /path/to/database_random/labels_euler.csv \
        --output /path/to/database_random/labels.csv
"""
import argparse
import csv
import math
from pathlib import Path


def euler_zyx_to_quaternion(
    pitch_deg: float, yaw_deg: float, roll_deg: float
) -> tuple[float, float, float, float]:
    p = math.radians(pitch_deg) / 2
    y = math.radians(yaw_deg) / 2
    r = math.radians(roll_deg) / 2
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)
    qw = cr * cy * cp + sr * sy * sp
    qx = cr * cy * sp - sr * sy * cp
    qy = cr * sy * cp + sr * cy * sp
    qz = sr * cy * cp - cr * sy * sp
    return qw, qx, qy, qz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="labels_euler.csv from MATLAB")
    parser.add_argument("--output", required=True, help="output labels.csv for train.py")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    count = 0
    with input_path.open(newline="") as fin, output_path.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        missing = {"image", "pitch_deg", "yaw_deg", "roll_deg"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"入力CSVに必要な列がありません: {sorted(missing)}")

        writer = csv.writer(fout)
        writer.writerow(["image", "qw", "qx", "qy", "qz"])

        for row in reader:
            qw, qx, qy, qz = euler_zyx_to_quaternion(
                float(row["pitch_deg"]),
                float(row["yaw_deg"]),
                float(row["roll_deg"]),
            )
            writer.writerow([
                row["image"],
                f"{qw:.8f}", f"{qx:.8f}", f"{qy:.8f}", f"{qz:.8f}",
            ])
            count += 1

    print(f"変換完了: {count}件 → {output_path}")


if __name__ == "__main__":
    main()
