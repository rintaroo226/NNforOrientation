"""距離が時間変化する軌道 (generate_trajectory_with_distance.py +
matlab/render_trajectory_with_distance.m の出力) に対して、bboxクロップモデル
の NN 推論 + EKF を一気通貫で走らせ、「遠方でノイジーな生の観測を EKF の
剛体力学モデルが平滑化できているか」を確認するスクリプト。

run_ekf_eval.py との違いは実質2点だけ:
  1. 画像の前処理が「そのままリサイズ」ではなく bboxcrop_dataset.bbox_crop_resize
     (train_bboxcrop.py で学習したモデルの入力と合わせるため)。
  2. labels.csv の distance 列を読み、誤差の時系列プロットに第2軸として
     重ねる (遠→近の遷移と誤差減少が対応しているかを目で見えるようにする)。
EKF 自体のロジック (run_ekf_pass) は run_ekf_eval.py と完全に同一 — EKF は
クォータニオン観測しか見ておらず distance を一切必要としないため、変更不要。
このプロジェクトの規約 (既存ファイルを改造するより新規ファイルを作り、
ヘルパーの重複を許容する) に従い、run_ekf_eval.py を import はせずこの
ファイル内に複製する。

Colab での使用例:
    !python run_ekf_eval_bboxcrop.py \
        --checkpoint checkpoints/silhouette_pose_bboxcrop.pt \
        --data-root  /content/database_trajectory_with_distance \
        --labels-csv /content/database_trajectory_with_distance/labels.csv
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

from bboxcrop_dataset import bbox_crop_resize
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
    p.add_argument("--checkpoint", required=True,
                    help="train_bboxcrop.py が保存したチェックポイント")
    p.add_argument("--data-root", required=True)
    p.add_argument("--labels-csv", required=True,
                    help="image,qw,qx,qy,qz,t,distance 形式のCSV "
                         "(matlab/render_trajectory_with_distance.m の出力)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="ekf_output_bboxcrop_distance")
    p.add_argument("--padding-factor", type=float, default=None,
                    help="省略時はチェックポイントに保存された値を使う "
                         "(train_bboxcrop.pyの--padding-factorと必ず一致させる)")
    p.add_argument("--box-size", default="3,2,1",
                    help="直方体サイズ 幅,高さ,奥行 [m] (generate_trajectory_with_distance.py と合わせる)")
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
                    help="初期共分散 (角速度, 対角成分)")
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


def load_image_tensor_bboxcrop(image_path: Path, image_size: int, padding_factor: float) -> torch.Tensor:
    """run_ekf_eval.load_image_tensor との唯一の差分: 全体リサイズではなく
    bbox_crop_resize (train_bboxcrop.py が学習時に使ったのと同じ前処理)。
    これが無いと、遠方フレームでは箱が画面の数%しか占めず、学習分布 (常に
    bboxで正規化された見かけの大きさ) と大きく食い違ってしまう。
    """
    arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    cropped = bbox_crop_resize(arr, image_size, padding_factor)
    return torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def load_trajectory_csv(path: Path) -> list[dict]:
    """image,t[,qw,qx,qy,qz][,distance] 形式の CSV を時刻順に読み込む。

    distance 列は EKF にもNNにも渡さない (bboxクロップがスケールを正規化する
    ため推論時に不要、EKFはクォータニオン観測しか見ない)。プロットで
    「遠方ほど誤差が大きい」ことを目で確認するためだけに保持する。
    """
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_gt = {"qw", "qx", "qy", "qz"}.issubset(fieldnames)
        has_distance = "distance" in fieldnames
        for row in reader:
            gt_q = None
            if has_gt:
                gt_q = np.array(
                    [row["qw"], row["qx"], row["qy"], row["qz"]], dtype=np.float32
                )
                gt_q /= max(float(np.linalg.norm(gt_q)), 1e-8)
            rows.append({
                "image": row["image"],
                "t": float(row["t"]),
                "gt_q": gt_q,
                "distance": float(row["distance"]) if has_distance else None,
            })
    rows.sort(key=lambda r: r["t"])
    return rows


def run_ekf_pass(
    rows: list[dict],
    pred_quats: np.ndarray,
    process_model: str,
    box_size: tuple[float, float, float],
    args: argparse.Namespace,
) -> dict:
    """run_ekf_eval.run_ekf_pass と完全に同一のロジック (EKFはdistanceを見ない)。"""
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


def estimate_initial_omega(pred_quats: np.ndarray, times: np.ndarray) -> np.ndarray:
    """run_ekf_eval.estimate_initial_omega と同一。"""
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
    padding_factor = args.padding_factor if args.padding_factor is not None else \
        float(ckpt.get("padding_factor", 1.25))
    model = SilhouettePoseNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"チェックポイント読み込み完了 (image_size={image_size}, "
          f"padding_factor={padding_factor}, best_angle_deg={ckpt.get('best_angle_deg')})")

    root = Path(args.data_root)
    rows = load_trajectory_csv(Path(args.labels_csv))
    has_gt = rows[0]["gt_q"] is not None
    has_distance = rows[0]["distance"] is not None
    n = len(rows)
    print(f"{n}フレームを時刻順に読み込みました (GTあり: {has_gt}, distanceあり: {has_distance})")

    # -----------------------------------------------------------------------
    # ① 全フレームをバッチ推論 (bboxクロップ前処理を適用)
    # -----------------------------------------------------------------------
    print("推論中 (bboxクロップ適用)...")
    infer_start = time.perf_counter()
    pred_quats = np.zeros((n, 4), dtype=np.float32)
    batch_size = 128
    for start in range(0, n, batch_size):
        batch = rows[start:start + batch_size]
        imgs = [load_image_tensor_bboxcrop(root / r["image"], image_size, padding_factor)
                for r in batch]
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
    # ② 時系列順に EKF を1ステップずつ適用 (run_ekf_eval.py と完全に同一ロジック)
    # -----------------------------------------------------------------------
    if args.omega0 is None:
        times_for_omega0 = np.array([r["t"] for r in rows])
        omega0_auto = estimate_initial_omega(pred_quats, times_for_omega0)
        args.omega0 = ",".join(str(v) for v in omega0_auto)
        print(f"ω0 を最初の2フレームから自動推定しました: {omega0_auto}")

    print(f"EKF 適用中 (process_model={args.process_model})...")
    result = run_ekf_pass(rows, pred_quats, args.process_model, box_size, args)
    ekf_quats = result["ekf_quats"]
    ekf_err = result["ekf_err"]
    accepted = result["accepted"]
    nis = result["nis"]
    sym_branch_idx = result["sym_branch_idx"]

    # -----------------------------------------------------------------------
    # 出力: CSV
    # -----------------------------------------------------------------------
    trace_path = out_dir / "ekf_trace.csv"
    with trace_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "t", "raw_err_deg", "ekf_err_deg", "accepted", "nis", "sym_branch_idx",
            "pred_qw", "pred_qx", "pred_qy", "pred_qz",
            "ekf_qw", "ekf_qx", "ekf_qy", "ekf_qz",
        ]
        if has_distance:
            header.append("distance")
        writer.writerow(header)
        for k in range(n):
            row = [
                f"{rows[k]['t']:.6f}",
                "" if math.isnan(raw_err[k]) else f"{raw_err[k]:.4f}",
                "" if math.isnan(ekf_err[k]) else f"{ekf_err[k]:.4f}",
                int(accepted[k]),
                f"{nis[k]:.4f}",
                sym_branch_idx[k],
                *[f"{v:.8f}" for v in pred_quats[k]],
                *[f"{v:.8f}" for v in ekf_quats[k]],
            ]
            if has_distance:
                row.append(f"{rows[k]['distance']:.4f}")
            writer.writerow(row)
    print(f"保存: {trace_path}")

    # -----------------------------------------------------------------------
    # 出力: 時系列プロット (誤差 + 第2軸に distance(t) を重ねる)
    # -----------------------------------------------------------------------
    if has_gt:
        t = np.array([r["t"] for r in rows])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, raw_err, label="生の NN 推定誤差 (bboxクロップ)", color="steelblue", alpha=0.7)
        ax.plot(t, ekf_err, label="EKF 事後誤差", color="tomato", linewidth=1.5)
        rejected = ~accepted
        if rejected.any():
            ax.scatter(t[rejected], raw_err[rejected], color="black", marker="x",
                       s=30, label="棄却された観測", zorder=5)
        ax.set_xlabel("時刻 [s]")
        ax.set_ylabel("対称性考慮角度誤差 [deg]")
        title = "距離変化軌道: 生のNN推定 vs EKF事後推定の角度誤差"

        if has_distance:
            distances = np.array([r["distance"] for r in rows])
            ax2 = ax.twinx()
            ax2.plot(t, distances, label="レンダリング距離", color="gray",
                     linestyle="--", linewidth=1.2, alpha=0.8)
            ax2.set_ylabel("レンダリング距離 [m]", color="gray")
            ax2.tick_params(axis="y", colors="gray")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            title += " (破線: 遠→近の距離遷移)"
        else:
            ax.legend()

        ax.set_title(title)
        fig.tight_layout()
        plot_path = out_dir / "ekf_error_vs_distance_timeseries.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"保存: {plot_path}")

        print("\n=== サマリ ===")
        print(f"  生の推定   平均: {np.nanmean(raw_err):.2f}°  中央値: {np.nanmedian(raw_err):.2f}°")
        print(f"  EKF事後推定 平均: {np.nanmean(ekf_err):.2f}°  中央値: {np.nanmedian(ekf_err):.2f}°")
        print(f"  棄却されたフレーム数: {int((~accepted).sum())}/{n}")

        if has_distance:
            # 遠方(前半)と近傍(後半)で分けて集計し、「遠方でのEKFの平滑化効果」を数値化する
            far_mask = distances >= np.median(distances)
            near_mask = ~far_mask
            for label, mask in (("遠方 (distance>=中央値)", far_mask), ("近傍 (distance<中央値)", near_mask)):
                if mask.any():
                    print(f"  {label}: 生の平均={np.nanmean(raw_err[mask]):.2f}°  "
                          f"EKF平均={np.nanmean(ekf_err[mask]):.2f}°  "
                          f"生の標準偏差={np.nanstd(raw_err[mask]):.2f}°  "
                          f"EKF標準偏差={np.nanstd(ekf_err[mask]):.2f}°")
    else:
        print("\nGT が無いため誤差プロット/サマリはスキップします。")

    print(f"\n完了。結果は {out_dir}/ に保存されました。")


if __name__ == "__main__":
    main()
