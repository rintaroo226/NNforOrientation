"""eval_vis.py - 学習済みモデルの予測結果を可視化する

Colab での使用例:
    !python eval_vis.py \
        --checkpoint checkpoints/silhouette_pose.pt \
        --data-root  /content/database_random \
        --labels-csv /content/database_random/labels.csv \
        --euler-csv  /content/database_random/labels_euler.csv
"""
import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from silhouette_pose.losses import (
    _BOX_SYMMETRIES,
    quat_mul,
    symmetry_aware_angle_error_deg,
)
from silhouette_pose.model import SilhouettePoseNet


# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data-root",   required=True)
    p.add_argument("--labels-csv",  required=True, help="クォータニオン CSV (train.py 用)")
    p.add_argument("--euler-csv",   default=None,  help="Euler CSV (任意, 元の pitch/yaw/roll)")
    p.add_argument("--device",      default="auto")
    p.add_argument("--n-samples",   type=int, default=16, help="グリッド表示サンプル数")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--out-dir",     default="eval_output")
    return p.parse_args()


# ---------------------------------------------------------------------------
# デバイス
# ---------------------------------------------------------------------------

def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------

def load_labels_csv(path: Path) -> dict[str, np.ndarray]:
    """image パスをキー, quaternion [4] を値とした辞書を返す。"""
    result: dict[str, np.ndarray] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = np.array([row["qw"], row["qx"], row["qy"], row["qz"]], dtype=np.float32)
            q /= max(float(np.linalg.norm(q)), 1e-8)
            result[row["image"]] = q
    return result


def load_euler_csv(path: Path) -> dict[str, tuple[float, float, float]]:
    """image パスをキー, (pitch, yaw, roll) [deg] を値とした辞書を返す。"""
    result: dict[str, tuple[float, float, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["image"]] = (
                float(row["pitch_deg"]),
                float(row["yaw_deg"]),
                float(row["roll_deg"]),
            )
    return result


def load_image_tensor(image_path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(image_path).convert("L").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


# ---------------------------------------------------------------------------
# クォータニオン → ZYX Euler 変換
# (R = Rz(roll) * Ry(yaw) * Rx(pitch) の逆変換)
# ---------------------------------------------------------------------------

def quat_to_euler_zyx_deg(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    pitch = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    sin_yaw = max(-1.0, min(1.0, 2*(w*y - z*x)))
    yaw   = math.degrees(math.asin(sin_yaw))
    roll  = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return pitch, yaw, roll


def best_sym_euler(pred_q: np.ndarray, target_q: np.ndarray) -> tuple[float, float, float]:
    """最も近い対称等価姿勢の Euler 角を返す。"""
    p = pred_q / max(float(np.linalg.norm(pred_q)), 1e-8)
    t = target_q / max(float(np.linalg.norm(target_q)), 1e-8)

    best_dot = -1.0
    best_sym_idx = 0
    for i, sym in enumerate(_BOX_SYMMETRIES.numpy()):
        equiv = _quat_mul_np(t, sym)
        d = abs(float(np.dot(p, equiv)))
        if d > best_dot:
            best_dot = d
            best_sym_idx = i

    sym = _BOX_SYMMETRIES[best_sym_idx].numpy()
    nearest_target = _quat_mul_np(t, sym)
    return quat_to_euler_zyx_deg(nearest_target)


def _quat_mul_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]
    return np.array([
        qw*rw - qx*rx - qy*ry - qz*rz,
        qw*rx + qx*rw + qy*rz - qz*ry,
        qw*ry - qx*rz + qy*rw + qz*rx,
        qw*rz + qx*ry - qy*rx + qz*rw,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # モデル読み込み
    ckpt = torch.load(args.checkpoint, map_location=device)
    image_size = int(ckpt.get("image_size", 64))
    model = SilhouettePoseNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint loaded. Best angle in training: {ckpt.get('best_angle_deg', '?'):.2f}°")

    # ラベル読み込み
    root = Path(args.data_root)
    labels = load_labels_csv(Path(args.labels_csv))
    euler_gt: dict[str, tuple[float, float, float]] | None = None
    if args.euler_csv:
        euler_gt = load_euler_csv(Path(args.euler_csv))

    all_keys = list(labels.keys())
    random.shuffle(all_keys)

    # -----------------------------------------------------------------------
    # 全サンプルで推論 → angle error を収集
    # -----------------------------------------------------------------------
    print(f"Running inference on {len(all_keys)} samples...")
    all_errors: list[float] = []
    all_pred_quats: dict[str, np.ndarray] = {}

    batch_size = 128
    for start in range(0, len(all_keys), batch_size):
        batch_keys = all_keys[start:start + batch_size]
        imgs, gts = [], []
        valid_keys = []
        for k in batch_keys:
            img_path = root / k
            if not img_path.exists():
                continue
            imgs.append(load_image_tensor(img_path, image_size))
            gts.append(torch.from_numpy(labels[k]))
            valid_keys.append(k)
        if not imgs:
            continue
        x = torch.cat(imgs, dim=0).to(device)
        y = torch.stack(gts).to(device)
        with torch.no_grad():
            pred = model(x)
        errors = symmetry_aware_angle_error_deg(pred, y).cpu().numpy()
        for k, p, e in zip(valid_keys, pred.cpu().numpy(), errors):
            all_errors.append(float(e))
            all_pred_quats[k] = p

    all_errors_arr = np.array(all_errors)
    print(f"\n=== 角度誤差統計 ===")
    print(f"  平均: {all_errors_arr.mean():.2f}°")
    print(f"  中央値: {np.median(all_errors_arr):.2f}°")
    print(f"  標準偏差: {all_errors_arr.std():.2f}°")
    print(f"  最小: {all_errors_arr.min():.2f}°")
    print(f"  最大: {all_errors_arr.max():.2f}°")
    print(f"  < 10°: {(all_errors_arr < 10).mean()*100:.1f}%")
    print(f"  < 20°: {(all_errors_arr < 20).mean()*100:.1f}%")
    print(f"  < 45°: {(all_errors_arr < 45).mean()*100:.1f}%")

    # -----------------------------------------------------------------------
    # 図1: 角度誤差ヒストグラム
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_errors_arr, bins=90, range=(0, 180), color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(all_errors_arr.mean(), color="tomato", linewidth=1.5, label=f"平均 {all_errors_arr.mean():.1f}°")
    ax.axvline(np.median(all_errors_arr), color="orange", linewidth=1.5, linestyle="--",
               label=f"中央値 {np.median(all_errors_arr):.1f}°")
    ax.set_xlabel("角度誤差 (°)")
    ax.set_ylabel("サンプル数")
    ax.set_title("対称性考慮済み角度誤差の分布")
    ax.legend()
    fig.tight_layout()
    hist_path = out_dir / "error_histogram.png"
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    print(f"\n保存: {hist_path}")

    # -----------------------------------------------------------------------
    # 図2: ランダムサンプルグリッド
    # -----------------------------------------------------------------------
    sample_keys = all_keys[:args.n_samples]
    n_cols = 4
    n_rows = math.ceil(args.n_samples / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.8))
    axes = axes.flatten()

    for ax, key in zip(axes, sample_keys):
        img_path = root / key
        if not img_path.exists():
            ax.axis("off")
            continue

        img = Image.open(img_path).convert("L")
        pred_q = all_pred_quats.get(key)
        gt_q   = labels[key]
        err    = float(all_errors_arr[all_keys.index(key)]) if key in all_keys else float("nan")

        ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

        # 真値 Euler 角
        if euler_gt and key in euler_gt:
            gp, gy, gr = euler_gt[key]
        else:
            gp, gy, gr = quat_to_euler_zyx_deg(gt_q)

        # 予測 Euler 角 (最も近い対称等価)
        if pred_q is not None:
            pp, py, pr = best_sym_euler(pred_q, gt_q)
            lines = [
                f"GT   P{gp:+6.1f}° Y{gy:+6.1f}° R{gr:+6.1f}°",
                f"Pred P{pp:+6.1f}° Y{py:+6.1f}° R{pr:+6.1f}°",
                f"Err  {err:.1f}°",
            ]
            color = "lime" if err < 20 else ("yellow" if err < 45 else "tomato")
        else:
            lines = [f"GT P{gp:+.1f}° Y{gy:+.1f}° R{gr:+.1f}°", "no pred"]
            color = "white"

        ax.set_title("\n".join(lines), fontsize=6.5, color=color,
                     fontfamily="monospace", loc="left",
                     bbox=dict(facecolor="black", alpha=0.65, pad=2))

    for ax in axes[args.n_samples:]:
        ax.axis("off")

    fig.suptitle("ランダムサンプル予測結果  (緑<20°  黄<45°  赤≥45°)", fontsize=10)
    fig.tight_layout()
    grid_path = out_dir / "sample_grid.png"
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)
    print(f"保存: {grid_path}")

    # -----------------------------------------------------------------------
    # 図3: ベスト5 / ワースト5
    # -----------------------------------------------------------------------
    sorted_keys = sorted(all_pred_quats.keys(), key=lambda k: all_errors_arr[all_keys.index(k)])
    best5  = sorted_keys[:5]
    worst5 = sorted_keys[-5:]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for row_idx, (row_keys, row_label) in enumerate([(best5, "Best 5"), (worst5, "Worst 5")]):
        for col_idx, key in enumerate(row_keys):
            ax = axes[row_idx, col_idx]
            img = Image.open(root / key).convert("L")
            ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255)
            ax.axis("off")

            gt_q   = labels[key]
            pred_q = all_pred_quats[key]
            err    = all_errors_arr[all_keys.index(key)]

            if euler_gt and key in euler_gt:
                gp, gy, gr = euler_gt[key]
            else:
                gp, gy, gr = quat_to_euler_zyx_deg(gt_q)
            pp, py, pr = best_sym_euler(pred_q, gt_q)

            ax.set_title(
                f"GT   P{gp:+.1f}° Y{gy:+.1f}° R{gr:+.1f}°\n"
                f"Pred P{pp:+.1f}° Y{py:+.1f}° R{pr:+.1f}°\n"
                f"Err {err:.1f}°",
                fontsize=7, fontfamily="monospace",
                bbox=dict(facecolor="black", alpha=0.7, pad=2),
                color="lime" if row_idx == 0 else "tomato",
            )

    axes[0, 0].set_ylabel("Best 5", fontsize=11, labelpad=40)
    axes[1, 0].set_ylabel("Worst 5", fontsize=11, labelpad=40)
    fig.suptitle("ベスト5・ワースト5 予測例", fontsize=12)
    fig.tight_layout()
    bw_path = out_dir / "best_worst.png"
    fig.savefig(bw_path, dpi=120)
    plt.close(fig)
    print(f"保存: {bw_path}")

    print(f"\n完了。結果は {out_dir}/ に保存されました。")


if __name__ == "__main__":
    main()
