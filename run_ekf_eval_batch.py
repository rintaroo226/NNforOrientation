"""複数本の軌道データセットに対して NN推論 + EKF (constant/euler) をまとめて走らせ、
統計的な効果検証を行うスクリプト。

matlab/render_trajectory_batch.m が出力する
`<data-root>/traj_000/{images/,labels.csv}`, `traj_001/...` という構造を想定する。

run_ekf_eval.py の主要ロジック (チェックポイント読み込み・EKF実行・反転検出) を
そのまま import して再利用し、複数軌道分をループしてから集計する。

Colab での使用例:
    !python run_ekf_eval_batch.py \
        --checkpoint checkpoints/silhouette_pose.pt \
        --data-root  /content/database_trajectory_batch \
        --out-dir    ekf_batch_output
"""
import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from quat_np import quat_conjugate, quat_mul, rotvec_from_quat, symmetry_aware_angle_error_deg
from run_ekf_eval import (
    choose_device,
    detect_flip_window,
    load_image_tensor,
    load_trajectory_csv,
    run_ekf_pass,
    setup_japanese_font,
)
from silhouette_pose.model import SilhouettePoseNet

setup_japanese_font()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True,
                    help="traj_000/, traj_001/, ... のサブフォルダを含むディレクトリ")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="ekf_batch_output")
    p.add_argument("--box-size", default="3,2,1")
    p.add_argument("--q-theta-rw", type=float, default=1e-5)
    p.add_argument("--q-omega-rw", type=float, default=1e-3)
    p.add_argument("--meas-noise-deg", type=float, default=2.0)
    p.add_argument("--gate-mode", choices=["chi2", "fixed", "both", "none"], default="both")
    p.add_argument("--gate-chi2-threshold", type=float, default=7.815)
    p.add_argument("--gate-max-angle-deg", type=float, default=30.0)
    p.add_argument("--p0-theta", type=float, default=0.5)
    p.add_argument("--p0-omega", type=float, default=2.0,
                    help="ω0 を最初の2フレームから概算するため、手動指定より不確実性を高めに取る")
    return p.parse_args()


def estimate_initial_omega(pred_quats: np.ndarray, times: np.ndarray) -> np.ndarray:
    """最初の2フレームの NN 観測から、有限差分で初期角速度を概算する (body frame)。

    真の初期角速度は実運用では分からない (各軌道でランダム) ので、EKF の外から
    真値を教えるのではなく、観測だけから求める。これをしないと ω0=0 のコールド
    スタートになり、速い回転の軌道で発散して統計がその発散に支配されてしまう。
    """
    dt0 = times[1] - times[0]
    rel = quat_mul(quat_conjugate(pred_quats[0]), pred_quats[1])
    if rel[0] < 0:
        rel = -rel
    return rotvec_from_quat(rel) / dt0


def run_one_trajectory(
    traj_dir: Path, model, device, image_size: int, box_size: tuple, args: argparse.Namespace,
) -> dict:
    rows = load_trajectory_csv(traj_dir / "labels.csv")
    has_gt = rows[0]["gt_q"] is not None
    n = len(rows)

    pred_quats = np.zeros((n, 4), dtype=np.float32)
    batch_size = 128
    for start in range(0, n, batch_size):
        batch = rows[start:start + batch_size]
        imgs = [load_image_tensor(traj_dir / r["image"], image_size) for r in batch]
        x = torch.cat(imgs, dim=0).to(device)
        with torch.no_grad():
            pred = model(x)
        pred_quats[start:start + len(batch)] = pred.cpu().numpy()

    raw_err = np.full(n, np.nan)
    if has_gt:
        for k in range(n):
            raw_err[k] = symmetry_aware_angle_error_deg(pred_quats[k], rows[k]["gt_q"])

    times = np.array([r["t"] for r in rows])
    omega0 = estimate_initial_omega(pred_quats, times)

    ekf_args = SimpleNamespace(
        q_theta_rw=args.q_theta_rw, q_omega_rw=args.q_omega_rw,
        meas_noise_deg=args.meas_noise_deg, gate_mode=args.gate_mode,
        gate_chi2_threshold=args.gate_chi2_threshold, gate_max_angle_deg=args.gate_max_angle_deg,
        omega0=",".join(str(v) for v in omega0),
        p0_theta=args.p0_theta, p0_omega=args.p0_omega,
    )

    result_const = run_ekf_pass(rows, pred_quats, "constant", box_size, ekf_args)
    result_euler = run_ekf_pass(rows, pred_quats, "euler", box_size, ekf_args)

    flip_t = None
    if has_gt:
        gt_quats = np.stack([r["gt_q"] for r in rows])
        flip_t = detect_flip_window(gt_quats, times, box_size)

    return {
        "n_frames": n,
        "has_gt": has_gt,
        "raw_mean": float(np.nanmean(raw_err)),
        "raw_median": float(np.nanmedian(raw_err)),
        "constant_mean": float(np.nanmean(result_const["ekf_err"])),
        "euler_mean": float(np.nanmean(result_euler["ekf_err"])),
        "constant_rejected": int((~result_const["accepted"]).sum()),
        "euler_rejected": int((~result_euler["accepted"]).sum()),
        "flip_t": flip_t,
        "omega0_est": omega0,
    }


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
    print(f"チェックポイント読み込み完了 (image_size={image_size})")

    data_root = Path(args.data_root)
    traj_dirs = sorted(
        d for d in data_root.iterdir() if d.is_dir() and (d / "labels.csv").exists()
    )
    print(f"{len(traj_dirs)}本の軌道を見つけました: {[d.name for d in traj_dirs]}")

    summary_rows = []
    start = time.perf_counter()
    for i, traj_dir in enumerate(traj_dirs):
        print(f"\n[{i+1}/{len(traj_dirs)}] {traj_dir.name} を処理中...")
        result = run_one_trajectory(traj_dir, model, device, image_size, box_size, args)
        result["traj"] = traj_dir.name
        summary_rows.append(result)
        print(f"  生={result['raw_mean']:.2f}°  constant={result['constant_mean']:.2f}°  "
              f"euler={result['euler_mean']:.2f}°  "
              f"棄却(constant/euler)={result['constant_rejected']}/{result['euler_rejected']}  "
              f"反転={result['flip_t']}")
    print(f"\n全軌道の処理時間: {time.perf_counter()-start:.1f}秒")

    # -----------------------------------------------------------------------
    # 出力: 集計 CSV
    # -----------------------------------------------------------------------
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["traj", "n_frames", "raw_mean", "raw_median",
                          "constant_mean", "euler_mean",
                          "constant_rejected", "euler_rejected", "flip_t"])
        for r in summary_rows:
            writer.writerow([
                r["traj"], r["n_frames"], f"{r['raw_mean']:.4f}", f"{r['raw_median']:.4f}",
                f"{r['constant_mean']:.4f}", f"{r['euler_mean']:.4f}",
                r["constant_rejected"], r["euler_rejected"],
                "" if r["flip_t"] is None else f"{r['flip_t']:.3f}",
            ])
    print(f"保存: {summary_path}")

    # -----------------------------------------------------------------------
    # 出力: 集計プロット (10本の軌道での分布比較)
    # -----------------------------------------------------------------------
    raw_means = [r["raw_mean"] for r in summary_rows]
    const_means = [r["constant_mean"] for r in summary_rows]
    euler_means = [r["euler_mean"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [raw_means, const_means, euler_means]
    labels = ["生のNN推定", "EKF (constant)", "EKF (euler)"]
    bp = ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for i, means in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.03, size=len(means))
        ax.scatter(np.full(len(means), i) + jitter, means, color="steelblue", alpha=0.6, zorder=3)
    ax.set_ylabel("軌道ごとの平均角度誤差 [deg]")
    ax.set_title(f"{len(summary_rows)}本の軌道での誤差分布 (各点=1軌道の平均誤差)")
    ax.set_yscale("log")
    fig.tight_layout()
    plot_path = out_dir / "batch_comparison.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"保存: {plot_path}")

    print("\n=== 全軌道の集計 ===")
    print(f"  生の推定    平均: {np.mean(raw_means):.2f}°  (軌道間の標準偏差: {np.std(raw_means):.2f}°)")
    print(f"  EKF constant 平均: {np.mean(const_means):.2f}°  (標準偏差: {np.std(const_means):.2f}°)")
    print(f"  EKF euler   平均: {np.mean(euler_means):.2f}°  (標準偏差: {np.std(euler_means):.2f}°)")
    n_euler_better = sum(e < r for e, r in zip(euler_means, raw_means))
    print(f"  euler が生推定より良かった軌道数: {n_euler_better}/{len(summary_rows)}")

    print(f"\n完了。結果は {out_dir}/ に保存されました。")


if __name__ == "__main__":
    main()
