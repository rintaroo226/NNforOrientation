"""run_ekf_eval.py の結果 (ekf_trace.csv) を、元のシルエット画像に
GT / NN生推定 / EKF事後推定 の3つのワイヤーフレームを重ねた動画として書き出す。

run_ekf_eval.py --out-dir ekf_output ... を実行した後、以下のように使う:

    python3 render_ekf_overlay.py \
        --data-root database_trajectory \
        --labels-csv database_trajectory/labels.csv \
        --ekf-trace ekf_output/ekf_trace.csv \
        --out-video ekf_overlay.mp4
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from PIL import Image

from quat_np import quat_to_matrix


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

# ---------------------------------------------------------------------------
# 直方体ワイヤーフレームの投影 (matlab/initBoxSim.m, renderBoxImage.m と同じ配置。
# eval_vis.py の同名ロジックを、既存ファイルを変更せずこのスクリプト用に
# 独立して再実装したもの)
# ---------------------------------------------------------------------------

_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # -Z 面
    (4, 5), (5, 6), (6, 7), (7, 4),  # +Z 面
    (0, 4), (1, 5), (2, 6), (3, 7),  # 縦の辺
]


def box_vertices(box_size: tuple[float, float, float]) -> np.ndarray:
    w, h, d = box_size
    return np.array([
        [-w/2, -h/2, -d/2], [w/2, -h/2, -d/2], [w/2, h/2, -d/2], [-w/2, h/2, -d/2],
        [-w/2, -h/2, d/2], [w/2, -h/2, d/2], [w/2, h/2, d/2], [-w/2, h/2, d/2],
    ], dtype=np.float32)


def project_box_edges(
    q: np.ndarray, box_size: tuple[float, float, float],
    distance: float, fov_deg: float, img_w: int, img_h: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    v = box_vertices(box_size) @ quat_to_matrix(q).T
    v[:, 2] += distance
    tan_half = math.tan(math.radians(fov_deg) / 2.0)
    # matlab/renderBoxImage.m のカメラ配置に合わせ、world +X を画面左に反転する
    # (eval_vis.py で確認済みの同じ補正)。
    ndc_x = -v[:, 0] / (v[:, 2] * tan_half)
    ndc_y = v[:, 1] / (v[:, 2] * tan_half)
    px = (ndc_x + 1.0) / 2.0 * img_w
    py = (1.0 - (ndc_y + 1.0) / 2.0) * img_h
    return [((px[i], py[i]), (px[j], py[j])) for i, j in _BOX_EDGES]


def draw_box_wireframe(ax, q, box_size, distance, fov_deg, img_w, img_h, **kwargs) -> None:
    for (x0, y0), (x1, y1) in project_box_edges(q, box_size, distance, fov_deg, img_w, img_h):
        ax.plot([x0, x1], [y0, y1], **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--labels-csv", required=True, help="image,qw,qx,qy,qz,t 形式 (GT付き)")
    p.add_argument("--ekf-trace", required=True,
                    help="run_ekf_eval.py が出力した ekf_trace.csv (pred/ekf クォータニオン列が必要)")
    p.add_argument("--out-video", default="ekf_overlay.mp4")
    p.add_argument("--fps", type=float, default=None,
                    help="出力動画のfps (未指定ならデータの時間刻みから自動計算)")
    p.add_argument("--box-size", default="3,2,1",
                    help="直方体サイズ 幅,高さ,奥行 [m] (render_trajectory.m と合わせる)")
    p.add_argument("--cam-distance", type=float, default=10.0)
    p.add_argument("--cam-fov", type=float, default=25.0)
    p.add_argument("--max-frames", type=int, default=None, help="動作確認用にフレーム数を制限する")
    return p.parse_args()


def load_labels_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_q = np.array([row["qw"], row["qx"], row["qy"], row["qz"]], dtype=np.float32)
            gt_q /= max(float(np.linalg.norm(gt_q)), 1e-8)
            rows.append({"image": row["image"], "t": float(row["t"]), "gt_q": gt_q})
    rows.sort(key=lambda r: r["t"])
    return rows


def load_ekf_trace(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_q = np.array([row["pred_qw"], row["pred_qx"], row["pred_qy"], row["pred_qz"]],
                               dtype=np.float32)
            ekf_q = np.array([row["ekf_qw"], row["ekf_qx"], row["ekf_qy"], row["ekf_qz"]],
                              dtype=np.float32)
            rows.append({
                "t": float(row["t"]),
                "pred_q": pred_q,
                "ekf_q": ekf_q,
                "accepted": bool(int(row["accepted"])),
                "raw_err_deg": float(row["raw_err_deg"]) if row["raw_err_deg"] else float("nan"),
                "ekf_err_deg": float(row["ekf_err_deg"]) if row["ekf_err_deg"] else float("nan"),
            })
    rows.sort(key=lambda r: r["t"])
    return rows


def main() -> None:
    args = parse_args()
    box_size = tuple(float(v) for v in args.box_size.split(","))

    label_rows = load_labels_csv(Path(args.labels_csv))
    trace_rows = load_ekf_trace(Path(args.ekf_trace))
    n = min(len(label_rows), len(trace_rows))
    if len(label_rows) != len(trace_rows):
        print(f"[警告] labels-csv ({len(label_rows)}行) と ekf-trace ({len(trace_rows)}行) の"
              f"行数が一致しません。先頭 {n} 行だけ使います。")
    if args.max_frames is not None:
        n = min(n, args.max_frames)

    ts_mismatch = np.abs(
        np.array([r["t"] for r in label_rows[:n]]) - np.array([r["t"] for r in trace_rows[:n]])
    )
    if ts_mismatch.max() > 1e-3:
        print(f"[警告] labels-csv と ekf-trace の時刻がずれています (最大差 {ts_mismatch.max():.4f}秒)。"
              f"両方とも同じ run_ekf_eval.py の実行から出たファイルか確認してください。")

    fps = args.fps
    if fps is None:
        dts = np.diff([r["t"] for r in trace_rows[:n]])
        fps = 1.0 / np.median(dts) if len(dts) > 0 else 30.0
    print(f"{n}フレームを {fps:.1f}fps で動画化します: {args.out_video}")

    root = Path(args.data_root)
    img0 = Image.open(root / label_rows[0]["image"])
    img_w, img_h = img0.size

    fig, ax = plt.subplots(figsize=(6, 6))
    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, args.out_video, dpi=120):
        for k in range(n):
            ax.clear()
            img = Image.open(root / label_rows[k]["image"]).convert("L")
            ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255)

            gt_q = label_rows[k]["gt_q"]
            pred_q = trace_rows[k]["pred_q"]
            ekf_q = trace_rows[k]["ekf_q"]

            draw_box_wireframe(ax, gt_q, box_size, args.cam_distance, args.cam_fov, img_w, img_h,
                                color="lime", linestyle="--", linewidth=1.3, label="GT")
            draw_box_wireframe(ax, pred_q, box_size, args.cam_distance, args.cam_fov, img_w, img_h,
                                color="orange", linestyle=":", linewidth=1.5, label="NN生推定")
            draw_box_wireframe(ax, ekf_q, box_size, args.cam_distance, args.cam_fov, img_w, img_h,
                                color="cyan", linestyle="-", linewidth=1.3, label="EKF事後推定")

            accepted = trace_rows[k]["accepted"]
            status = "採用" if accepted else "棄却"
            status_color = "white" if accepted else "red"
            ax.set_title(
                f"t={trace_rows[k]['t']:.2f}s  "
                f"生誤差={trace_rows[k]['raw_err_deg']:.1f}°  "
                f"EKF誤差={trace_rows[k]['ekf_err_deg']:.1f}°  "
                f"観測: {status}",
                fontsize=9, color=status_color,
                bbox=dict(facecolor="black", alpha=0.6, pad=3),
            )
            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)
            ax.axis("off")
            if k == 0:
                ax.legend(loc="upper right", fontsize=8, facecolor="black", labelcolor="white")

            writer.grab_frame()

    plt.close(fig)
    print(f"完了。動画を保存しました: {args.out_video}")


if __name__ == "__main__":
    main()
