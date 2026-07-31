"""現行モデル(distance=10固定で学習)が、距離を変えた入力に対してどの程度
姿勢推定精度が落ちるかを実測するスクリプト。

matlab/generate_distance_sweep_database.m が出力する
`image,qw,qx,qy,qz,distance` 形式の labels.csv を読み、距離ごとに
角度誤差の平均・中央値・分布を集計・プロットする。

前処理は eval_vis.py/run_ekf_eval.py と同じ「画像全体を image_size に
リサイズするだけ(bboxクロップ無し)」を使う。これは今のモデルの実際の
推論パイプラインそのものなので、"bboxクロップを導入する前の現状の弱点"
を素の数値として測ることが目的。

Colab での使用例:
    !python eval_distance_sweep.py \
        --checkpoint checkpoints/silhouette_pose.pt \
        --data-root  /content/database_distance_sweep \
        --labels-csv /content/database_distance_sweep/labels.csv
"""
import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from quat_np import symmetry_aware_angle_error_deg
from silhouette_pose.model import SilhouettePoseNet


def setup_japanese_font() -> None:
    """matplotlib のグラフ内日本語が □ 化けするのを防ぐフォント設定。"""
    try:
        import japanize_matplotlib  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import subprocess
        import sys
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "japanize-matplotlib"],
            check=True,
        )
        import japanize_matplotlib  # noqa: F401
        print("[情報] japanize-matplotlib を自動インストールしました。")
        return
    except Exception:
        pass
    candidates = [
        "Hiragino Sans", "Hiragino Kaku Gothic ProN",
        "Yu Gothic", "Meiryo", "MS Gothic",
        "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    print(
        "[警告] 日本語フォントの自動設定に失敗しました。グラフの日本語が □ になる場合は "
        "`pip install japanize-matplotlib` を手動で実行してから再実行してください。"
    )


setup_japanese_font()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--labels-csv", required=True,
                    help="image,qw,qx,qy,qz,distance 形式のCSV "
                         "(matlab/generate_distance_sweep_database.m の出力)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="distance_sweep_output")
    p.add_argument("--training-distance", type=float, default=10.0,
                    help="グラフに基準線として表示する学習時の距離")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image_tensor(image_path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(image_path).convert("L").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def load_labels_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_q = np.array(
                [row["qw"], row["qx"], row["qy"], row["qz"]], dtype=np.float32
            )
            gt_q /= max(float(np.linalg.norm(gt_q)), 1e-8)
            rows.append({
                "image": row["image"],
                "gt_q": gt_q,
                "distance": float(row["distance"]),
            })
    return rows


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    image_size = int(ckpt.get("image_size", 64))
    model = SilhouettePoseNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"チェックポイント読み込み完了 (image_size={image_size})")

    root = Path(args.data_root)
    rows = load_labels_csv(Path(args.labels_csv))
    n = len(rows)
    print(f"{n}枚を読み込みました")

    print("推論中...")
    infer_start = time.perf_counter()
    pred_quats = np.zeros((n, 4), dtype=np.float32)
    batch_size = 128
    for start in range(0, n, batch_size):
        batch = rows[start:start + batch_size]
        imgs = [load_image_tensor(root / r["image"], image_size) for r in batch]
        x = torch.cat(imgs, dim=0).to(device)
        with torch.no_grad():
            pred = model(x)
        pred_quats[start:start + len(batch)] = pred.cpu().numpy()
    infer_elapsed = time.perf_counter() - infer_start
    print(f"推論時間: {infer_elapsed:.1f}秒 ({n}枚, {n/infer_elapsed:.1f}枚/秒)")

    err = np.array([
        symmetry_aware_angle_error_deg(pred_quats[i], rows[i]["gt_q"]) for i in range(n)
    ])
    distances = np.array([r["distance"] for r in rows])
    uniq_dist = sorted(set(distances.tolist()))

    # -----------------------------------------------------------------------
    # 出力: 距離ごとの集計CSV
    # -----------------------------------------------------------------------
    means, medians, p90s = [], [], []
    print("\n=== 距離ごとの角度誤差 ===")
    for d in uniq_dist:
        e = err[distances == d]
        means.append(float(e.mean()))
        medians.append(float(np.median(e)))
        p90s.append(float(np.percentile(e, 90)))
        print(f"  distance={d:6.1f}  平均={e.mean():6.2f}°  中央値={np.median(e):6.2f}°  "
              f"90%ile={np.percentile(e, 90):6.2f}°  (n={int((distances == d).sum())})")

    summary_path = out_dir / "distance_sweep_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance", "mean_err_deg", "median_err_deg", "p90_err_deg", "n"])
        for d, m, med, p90 in zip(uniq_dist, means, medians, p90s):
            w.writerow([d, f"{m:.4f}", f"{med:.4f}", f"{p90:.4f}", int((distances == d).sum())])
    print(f"保存: {summary_path}")

    # -----------------------------------------------------------------------
    # 出力: 誤差 vs 距離の折れ線プロット
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(uniq_dist, means, marker="o", label="平均誤差")
    ax.plot(uniq_dist, medians, marker="s", label="中央値誤差")
    ax.plot(uniq_dist, p90s, marker="^", linestyle="--", label="90%ile誤差")
    ax.axvline(args.training_distance, color="gray", linestyle=":", label="学習時の距離")
    ax.set_xlabel("レンダリング距離")
    ax.set_ylabel("対称性考慮角度誤差 [deg]")
    ax.set_title("距離変化に対する現行モデルの姿勢推定精度")
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "distance_sweep_error.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"保存: {plot_path}")

    # -----------------------------------------------------------------------
    # 出力: 距離ごとの誤差分布(箱ひげ図)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [err[distances == d] for d in uniq_dist]
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(uniq_dist) + 1))
    ax.set_xticklabels([f"{d:.0f}" for d in uniq_dist])
    ax.set_xlabel("レンダリング距離")
    ax.set_ylabel("対称性考慮角度誤差 [deg]")
    ax.set_title("距離ごとの誤差分布")
    ax.set_yscale("log")
    fig.tight_layout()
    box_path = out_dir / "distance_sweep_boxplot.png"
    fig.savefig(box_path, dpi=120)
    plt.close(fig)
    print(f"保存: {box_path}")

    print(f"\n完了。結果は {out_dir}/ に保存されました。")


if __name__ == "__main__":
    main()
