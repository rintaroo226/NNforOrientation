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
from quat_np import quat_conjugate, quat_mul, rotvec_from_quat, symmetry_aware_angle_error_deg
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
    p.add_argument("--omega0", default=None,
                    help="EKF の初期角速度推定値 wx,wy,wz [rad/s] (未指定なら最初の2フレームの"
                         "NN観測から有限差分で自動推定する)")
    p.add_argument("--p0-theta", type=float, default=0.5, help="初期共分散 (姿勢, 対角成分)")
    p.add_argument("--p0-omega", type=float, default=2.0,
                    help="初期共分散 (角速度, 対角成分)。ω0 を自動推定する場合、"
                         "手動指定より不確実性が高いため既定値を大きめにしている")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare-euler", action="store_true",
                    help="--process-model の他にもう一方のモデルも走らせ、"
                         "ジャニベコフ的な反転区間を拡大した比較プロットを追加で出力する")
    p.add_argument("--zoom-window-sec", type=float, default=1.5,
                    help="反転区間の拡大表示で前後に見せる時間幅 [秒]")
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


def run_ekf_pass(
    rows: list[dict],
    pred_quats: np.ndarray,
    process_model: str,
    box_size: tuple[float, float, float],
    args: argparse.Namespace,
) -> dict:
    """rows/pred_quats に対して1つの運動モデルで EKF を時系列順に走らせる。

    --compare-euler で constant/euler の両方を同じ観測列に対して走らせる際に、
    推論をやり直さず (pred_quats は共有) この関数だけを2回呼べば済むようにする。
    """
    n = len(rows)
    inertia = principal_inertia(box_size) if process_model == "euler" else None
    config = EKFConfig(
        process_model=process_model,
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
    ekf_quats = np.zeros((n, 4), dtype=np.float64)
    ekf_err = np.full(n, np.nan)
    accepted = np.zeros(n, dtype=bool)
    nis = np.zeros(n)
    sym_branch_idx = np.zeros(n, dtype=int)

    ekf_quats[0] = pred_quats[0]
    accepted[0] = True
    has_gt = rows[0]["gt_q"] is not None
    if has_gt:
        ekf_err[0] = symmetry_aware_angle_error_deg(pred_quats[0], rows[0]["gt_q"])

    for k in range(1, n):
        dt = rows[k]["t"] - rows[k - 1]["t"]
        ekf.predict(dt)
        result = ekf.update(pred_quats[k])
        q_est, _, _ = ekf.state()
        ekf_quats[k] = q_est
        accepted[k] = result.accepted
        nis[k] = result.nis
        sym_branch_idx[k] = result.sym_branch_idx
        if has_gt:
            ekf_err[k] = symmetry_aware_angle_error_deg(q_est, rows[k]["gt_q"])

    return {
        "ekf_quats": ekf_quats,
        "ekf_err": ekf_err,
        "accepted": accepted,
        "nis": nis,
        "sym_branch_idx": sym_branch_idx,
    }


def detect_flip_window(
    gt_quats: np.ndarray, times: np.ndarray, box_size: tuple[float, float, float],
) -> float | None:
    """GT クォータニオン列だけから中間主軸まわりの角速度を有限差分で概算し、
    その符号が反転する最初の時刻を返す (ジャニベコフ的フリップの検出)。

    generate_trajectory.py の wx,wy,wz 列は matlab/render_trajectory.m の
    labels.csv には含まれない (画像レンダリングに不要なため) ので、GT の
    クォータニオン列から q(k)^-1 ⊗ q(k+1) の log map として角速度を復元する。
    見つからなければ None を返す。
    """
    inertia = principal_inertia(box_size)
    mid_axis = int(np.argsort(inertia)[1])  # 慣性モーメントが中間の軸 (不安定軸)

    n = len(gt_quats)
    omega_mid = np.zeros(n)
    for k in range(n - 1):
        dt = times[k + 1] - times[k]
        if dt <= 0:
            continue
        rel = quat_mul(quat_conjugate(gt_quats[k]), gt_quats[k + 1])
        if rel[0] < 0:
            rel = -rel
        omega_mid[k] = rotvec_from_quat(rel)[mid_axis] / dt
    omega_mid[-1] = omega_mid[-2] if n > 1 else 0.0

    signs = np.sign(omega_mid)
    flip_indices = np.where(np.diff(signs) != 0)[0]
    if len(flip_indices) == 0:
        return None
    return float(times[flip_indices[0]])


def estimate_initial_omega(pred_quats: np.ndarray, times: np.ndarray) -> np.ndarray:
    """最初の2フレームの NN 観測から、有限差分で初期角速度を概算する (body frame)。

    真の初期角速度は実運用では分からない (箱を放り投げるたびランダム) ので、
    EKF の外から真値を教えるのではなく観測だけから求める。これをしないと
    ω0=0 のコールドスタートになり、速い回転の軌道で発散するリスクがある。
    """
    dt0 = times[1] - times[0]
    rel = quat_mul(quat_conjugate(pred_quats[0]), pred_quats[1])
    if rel[0] < 0:
        rel = -rel
    return rotvec_from_quat(rel) / dt0


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
    # ② 時系列順に EKF を1ステップずつ適用 (--process-model で指定した方が主系列)
    # -----------------------------------------------------------------------
    if args.omega0 is None:
        times_for_omega0 = np.array([r["t"] for r in rows])
        omega0_auto = estimate_initial_omega(pred_quats, times_for_omega0)
        args.omega0 = ",".join(str(v) for v in omega0_auto)
        print(f"ω0 を最初の2フレームから自動推定しました: {omega0_auto}")

    print(f"EKF 適用中 (process_model={args.process_model})...")
    primary = run_ekf_pass(rows, pred_quats, args.process_model, box_size, args)
    ekf_quats = primary["ekf_quats"]
    ekf_err = primary["ekf_err"]
    accepted = primary["accepted"]
    nis = primary["nis"]
    sym_branch_idx = primary["sym_branch_idx"]

    # --compare-euler: もう一方のモデルも同じ pred_quats に対して走らせ、
    # 比較プロット用の誤差系列だけを追加で得る (推論はやり直さない)
    compare_result = None
    if args.compare_euler:
        other_model = "constant" if args.process_model == "euler" else "euler"
        print(f"比較用に process_model={other_model} でも EKF を走らせています...")
        compare_result = run_ekf_pass(rows, pred_quats, other_model, box_size, args)

    # -----------------------------------------------------------------------
    # 出力: CSV (pred/ekf クォータニオンも含める。動画オーバーレイ等で再利用するため)
    # -----------------------------------------------------------------------
    trace_path = out_dir / "ekf_trace.csv"
    with trace_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t", "raw_err_deg", "ekf_err_deg", "accepted", "nis", "sym_branch_idx",
            "pred_qw", "pred_qx", "pred_qy", "pred_qz",
            "ekf_qw", "ekf_qx", "ekf_qy", "ekf_qz",
        ])
        for k in range(n):
            writer.writerow([
                f"{rows[k]['t']:.6f}",
                "" if math.isnan(raw_err[k]) else f"{raw_err[k]:.4f}",
                "" if math.isnan(ekf_err[k]) else f"{ekf_err[k]:.4f}",
                int(accepted[k]),
                f"{nis[k]:.4f}",
                sym_branch_idx[k],
                *[f"{v:.8f}" for v in pred_quats[k]],
                *[f"{v:.8f}" for v in ekf_quats[k]],
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

        # -------------------------------------------------------------------
        # --compare-euler: constant/euler 比較 + 反転区間の拡大プロット
        # -------------------------------------------------------------------
        if compare_result is not None:
            other_model = "constant" if args.process_model == "euler" else "euler"
            other_err = compare_result["ekf_err"]
            gt_quats = np.stack([r["gt_q"] for r in rows])
            flip_t = detect_flip_window(gt_quats, t, box_size)

            labels = {args.process_model: ekf_err, other_model: other_err}
            n_rows_fig = 2 if flip_t is not None else 1
            fig, axes = plt.subplots(n_rows_fig, 1, figsize=(10, 4 * n_rows_fig), squeeze=False)
            axes = axes[:, 0]

            ax = axes[0]
            ax.plot(t, raw_err, label="生の NN 推定誤差", color="gray", alpha=0.6)
            for model_name, err in labels.items():
                ax.plot(t, err, label=f"EKF事後誤差 ({model_name})", linewidth=1.5)
            if flip_t is not None:
                ax.axvspan(flip_t - args.zoom_window_sec, flip_t + args.zoom_window_sec,
                           color="yellow", alpha=0.15, label="下段の拡大区間")
            ax.set_xlabel("時刻 [s]")
            ax.set_ylabel("対称性考慮角度誤差 [deg]")
            ax.set_title(f"運動モデル比較: {args.process_model} vs {other_model} (全区間)")
            ax.legend()

            if flip_t is not None:
                ax2 = axes[1]
                mask = (t >= flip_t - args.zoom_window_sec) & (t <= flip_t + args.zoom_window_sec)
                ax2.plot(t[mask], raw_err[mask], label="生の NN 推定誤差", color="gray", alpha=0.6)
                for model_name, err in labels.items():
                    ax2.plot(t[mask], err[mask], label=f"EKF事後誤差 ({model_name})",
                             linewidth=1.5, marker="o", markersize=2)
                ax2.axvline(flip_t, color="black", linestyle="--", linewidth=1, alpha=0.7)
                ax2.set_xlabel("時刻 [s]")
                ax2.set_ylabel("対称性考慮角度誤差 [deg]")
                ax2.set_title(f"反転区間の拡大 (t≈{flip_t:.2f}s ± {args.zoom_window_sec:.1f}s)")
                ax2.legend()
                print(f"  検出した反転時刻: t≈{flip_t:.2f}s")
            else:
                print("  反転区間は検出されませんでした (拡大パネルはスキップ)")

            fig.tight_layout()
            compare_path = out_dir / "ekf_model_comparison.png"
            fig.savefig(compare_path, dpi=120)
            plt.close(fig)
            print(f"保存: {compare_path}")
            print(f"  {args.process_model} 平均: {np.nanmean(ekf_err):.2f}°  "
                  f"{other_model} 平均: {np.nanmean(other_err):.2f}°")
    else:
        print("\nGT が無いため誤差プロット/サマリはスキップします "
              "(ekf_trace.csv に事後クォータニオンではなく誤差列は空欄で出力されます)。")

    print(f"\n完了。結果は {out_dir}/ に保存されました。")


if __name__ == "__main__":
    main()
