"""合成/実軌道データセットに対して NN 推論 + EKF を一気通貫で走らせ、
外れ値抑制の効果を確認するスクリプト。

想定する labels CSV は generate_trajectory.py + matlab/render_trajectory.m が
出力する `image,qw,qx,qy,qz,t` 形式。ただし qw..qz 列が無くても動作する
(実動画由来の `image,t` だけの CSV を将来渡せるようにするための設計)。

Colab での使用例:
    !python run_ekf_eval.py \
        --checkpoint checkpoints/silhouette_pose.pt \
        --data-root  /content/database_trajectory \
        --labels-csv /content/database_trajectory/labels.csv
"""
import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ekf import BoxOrientationEKF, EKFConfig
from quat_np import symmetry_aware_angle_error_deg
from rigid_body import principal_inertia
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
                    help="image,t[,qw,qx,qy,qz] 形式の CSV (GT列は無くても動作する)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="ekf_output")
    p.add_argument("--box-size", default="3,2,1",
                    help="直方体サイズ 幅,高さ,奥行 [m] (generate_trajectory.py と合わせる)")
    p.add_argument("--process-model", choices=["constant", "euler"], default="constant",
                    help="EKF の予測ステップで使う運動モデル")
    p.add_argument("--q-theta-rw", type=float, default=1e-5, help="姿勢誤差のプロセスノイズ [rad^2/s]")
    p.add_argument("--q-omega-rw", type=float, default=1e-3, help="角速度のプロセスノイズ [(rad/s)^2/s]")
    p.add_argument("--meas-noise-deg", type=float, default=2.0, help="観測(NN予測)誤差の標準偏差 [deg]")
    p.add_argument("--gate-mode", choices=["chi2", "fixed", "both", "none"], default="both")
    p.add_argument("--gate-chi2-threshold", type=float, default=7.815)
    p.add_argument("--gate-max-angle-deg", type=float, default=30.0)
    p.add_argument("--omega0", default="0,0,0", help="EKF の初期角速度推定値 wx,wy,wz [rad/s]")
    p.add_argument("--p0-theta", type=float, default=0.5, help="初期共分散 (姿勢, 対角成分)")
    p.add_argument("--p0-omega", type=float, default=1.0, help="初期共分散 (角速度, 対角成分)")
    p.add_argument("--seed", type=int, default=0)
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


def load_trajectory_csv(path: Path) -> list[dict]:
    """image,t[,qw,qx,qy,qz] 形式の CSV を時刻順に読み込む。

    SilhouettePoseDataset はシャッフルされたランダムアクセス学習用なので、
    時系列を厳密に保つ必要があるここでは使わず、専用のローダーを用意する。
    """
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        has_gt = {"qw", "qx", "qy", "qz"}.issubset(reader.fieldnames or [])
        for row in reader:
            gt_q = None
            if has_gt:
                gt_q = np.array(
                    [row["qw"], row["qx"], row["qy"], row["qz"]], dtype=np.float32
                )
                gt_q /= max(float(np.linalg.norm(gt_q)), 1e-8)
            rows.append({"image": row["image"], "t": float(row["t"]), "gt_q": gt_q})
    rows.sort(key=lambda r: r["t"])
    return rows


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    box_size = tuple(float(v) for v in args.box_size.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    image_size = int(ckpt.get("image_size", 64))
    model = SilhouettePoseNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"チェックポイント読み込み完了 (image_size={image_size}, "
          f"best_angle_deg={ckpt.get('best_angle_deg')})")

    root = Path(args.data_root)
    rows = load_trajectory_csv(Path(args.labels_csv))
    has_gt = rows[0]["gt_q"] is not None
    n = len(rows)
    print(f"{n}フレームを時刻順に読み込みました (GTあり: {has_gt})")

    # -----------------------------------------------------------------------
    # ① 全フレームをバッチ推論
    # -----------------------------------------------------------------------
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

    raw_err = np.full(n, np.nan)
    if has_gt:
        for k in range(n):
            raw_err[k] = symmetry_aware_angle_error_deg(pred_quats[k], rows[k]["gt_q"])

    # -----------------------------------------------------------------------
    # ② 時系列順に EKF を1ステップずつ適用
    # -----------------------------------------------------------------------
    print("EKF 適用中...")
    inertia = principal_inertia(box_size) if args.process_model == "euler" else None
    config = EKFConfig(
        process_model=args.process_model,
        inertia=inertia,
        q_theta_rw=args.q_theta_rw,
        q_omega_rw=args.q_omega_rw,
        meas_noise_deg=args.meas_noise_deg,
        gate_mode=args.gate_mode,
        gate_chi2_threshold=args.gate_chi2_threshold,
        gate_max_angle_deg=args.gate_max_angle_deg,
    )
    omega0 = np.array([float(v) for v in args.omega0.split(",")])
    P0 = np.eye(6)
    P0[:3, :3] *= args.p0_theta
    P0[3:, 3:] *= args.p0_omega

    ekf = BoxOrientationEKF(pred_quats[0], omega0, P0, config)
    ekf_err = np.full(n, np.nan)
    accepted = np.zeros(n, dtype=bool)
    nis = np.zeros(n)
    sym_branch_idx = np.zeros(n, dtype=int)
    if has_gt:
        ekf_err[0] = symmetry_aware_angle_error_deg(pred_quats[0], rows[0]["gt_q"])
    accepted[0] = True

    for k in range(1, n):
        dt = rows[k]["t"] - rows[k - 1]["t"]
        ekf.predict(dt)
        result = ekf.update(pred_quats[k])
        q_est, _, _ = ekf.state()
        accepted[k] = result.accepted
        nis[k] = result.nis
        sym_branch_idx[k] = result.sym_branch_idx
        if has_gt:
            ekf_err[k] = symmetry_aware_angle_error_deg(q_est, rows[k]["gt_q"])

    # -----------------------------------------------------------------------
    # 出力: CSV
    # -----------------------------------------------------------------------
    trace_path = out_dir / "ekf_trace.csv"
    with trace_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "raw_err_deg", "ekf_err_deg", "accepted", "nis", "sym_branch_idx"])
        for k in range(n):
            writer.writerow([
                f"{rows[k]['t']:.6f}",
                "" if math.isnan(raw_err[k]) else f"{raw_err[k]:.4f}",
                "" if math.isnan(ekf_err[k]) else f"{ekf_err[k]:.4f}",
                int(accepted[k]),
                f"{nis[k]:.4f}",
                sym_branch_idx[k],
            ])
    print(f"保存: {trace_path}")

    # -----------------------------------------------------------------------
    # 出力: 時系列プロット (GT がある場合のみ意味のある誤差プロットになる)
    # -----------------------------------------------------------------------
    if has_gt:
        fig, ax = plt.subplots(figsize=(10, 4))
        t = np.array([r["t"] for r in rows])
        ax.plot(t, raw_err, label="生の NN 推定誤差", color="steelblue", alpha=0.7)
        ax.plot(t, ekf_err, label="EKF 事後誤差", color="tomato", linewidth=1.5)
        rejected = ~accepted
        if rejected.any():
            ax.scatter(t[rejected], raw_err[rejected], color="black", marker="x",
                       s=30, label="棄却された観測", zorder=5)
        ax.set_xlabel("時刻 [s]")
        ax.set_ylabel("対称性考慮角度誤差 [deg]")
        ax.set_title("生のNN推定 vs EKF事後推定の角度誤差")
        ax.legend()
        fig.tight_layout()
        plot_path = out_dir / "ekf_error_timeseries.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"保存: {plot_path}")

        print("\n=== サマリ ===")
        print(f"  生の推定   平均: {np.nanmean(raw_err):.2f}°  中央値: {np.nanmedian(raw_err):.2f}°")
        print(f"  EKF事後推定 平均: {np.nanmean(ekf_err):.2f}°  中央値: {np.nanmedian(ekf_err):.2f}°")
        print(f"  棄却されたフレーム数: {int((~accepted).sum())}/{n}")
    else:
        print("\nGT が無いため誤差プロット/サマリはスキップします "
              "(ekf_trace.csv に事後クォータニオンではなく誤差列は空欄で出力されます)。")

    print(f"\n完了。結果は {out_dir}/ に保存されました。")


if __name__ == "__main__":
    main()
