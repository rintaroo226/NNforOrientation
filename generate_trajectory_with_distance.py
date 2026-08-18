"""distance が時間変化する剛体回転軌道を生成する (torch/MATLAB不要)。

generate_trajectory.py は render_distance を固定 (matlab/render_trajectory.m 側の
定数 ref_distance) して回転だけを積分するが、このスクリプトは「箱が回転しながら
カメラに接近してくる」ような、距離が時間とともに変化するシナリオを検証するために
distance(t) の列を追加した軌道を出力する。

回転力学 (rigid_body.integrate_trajectory) と距離プロファイルは物理的に無関係
(このプロジェクトには並進の運動方程式が無く、独立な運動学プロファイルとして
distance(t) を与えるだけ) なので、rigid_body.py には一切手を加えず、既存の
integrate_trajectory をそのまま呼び出す。distance 列は事後的に付け足すだけ。

出力 CSV (t,qw,qx,qy,qz,wx,wy,wz,distance) は、
matlab/render_trajectory_with_distance.m に渡してフレームごとに異なる
pose.distance でレンダリングするための入力になる。

実行例:
    python3 generate_trajectory_with_distance.py --out-csv trajectory_approach.csv \
        --duration 8 --fps 60 --process-model euler \
        --far-distance 55 --near-distance 5 --distance-profile ease
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from quat_np import sample_uniform_quat
from rigid_body import integrate_trajectory, principal_inertia


def distance_profile(t: np.ndarray, duration: float, far: float, near: float, profile: str) -> np.ndarray:
    """時刻配列 t ([0, duration]) に対応する distance(t) を返す。

    profile="linear": far→near に一定速度で接近 (最も単純、速度不連続は無いが
                       加速度が t=0 で不連続に生じる)
    profile="ease":    smoothstep (3s^2-2s^3) を使い、開始・終端で速度が 0 に
                        なめらかに漸近する接近 (カメラへの「ドッキング」的な
                        減速接近を模したい場合に使う)
    profile="constant": distance=far で一定 (distance(t) 機構自体の回帰テスト用。
                         generate_trajectory.py + 固定 ref_distance と等価になるはず)
    """
    s = np.clip(t / duration, 0.0, 1.0) if duration > 0 else np.zeros_like(t)
    if profile == "linear":
        shape = s
    elif profile == "ease":
        shape = 3 * s**2 - 2 * s**3
    elif profile == "constant":
        shape = np.zeros_like(s)
    else:
        raise ValueError(f"unknown distance profile: {profile!r}")
    return far + (near - far) * shape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-csv", required=True)
    p.add_argument("--duration", type=float, default=5.0, help="軌道の長さ [秒]")
    p.add_argument("--fps", type=float, default=60.0, help="フレームレート [Hz]")
    p.add_argument("--box-size", default="3,2,1",
                    help="直方体サイズ 幅,高さ,奥行 [m] (generate_random_database.m の target_size と合わせる)")
    p.add_argument("--process-model", choices=["constant", "euler"], default="euler",
                    help="運動モデル (constant: ω一定, euler: 無トルク剛体の運動方程式)")
    p.add_argument("--omega0", default=None,
                    help="初期角速度 wx,wy,wz [rad/s] (未指定ならランダム。任意方向を許容する)")
    p.add_argument("--omega0-mag", type=float, default=3.0,
                    help="--omega0 未指定時のランダム角速度の大きさ [rad/s]")
    p.add_argument("--q0", default=None,
                    help="初期姿勢 qw,qx,qy,qz (未指定なら SO(3) 上一様サンプリング)")
    p.add_argument("--far-distance", type=float, default=55.0,
                    help="軌道開始時のレンダリング距離 [m] (eval_distance_sweep* の遠方端に合わせた既定値)")
    p.add_argument("--near-distance", type=float, default=5.0,
                    help="軌道終了時のレンダリング距離 [m] (eval_distance_sweep* の近傍端に合わせた既定値)")
    p.add_argument("--distance-profile", choices=["linear", "ease", "constant"], default="linear",
                    help="distance(t) の時間プロファイル")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    box_size = tuple(float(v) for v in args.box_size.split(","))
    inertia = principal_inertia(box_size)

    if args.q0 is not None:
        q0 = np.array([float(v) for v in args.q0.split(",")])
    else:
        q0 = sample_uniform_quat(rng).astype(np.float64)

    if args.omega0 is not None:
        omega0 = np.array([float(v) for v in args.omega0.split(",")])
    else:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        omega0 = axis * args.omega0_mag

    dt = 1.0 / args.fps
    n_steps = int(round(args.duration * args.fps))

    times, quats, omegas = integrate_trajectory(q0, omega0, dt, n_steps, inertia, args.process_model)
    distances = distance_profile(times, args.duration, args.far_distance, args.near_distance,
                                  args.distance_profile)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "qw", "qx", "qy", "qz", "wx", "wy", "wz", "distance"])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.6f}",
                *[f"{v:.8f}" for v in quats[i]],
                *[f"{v:.8f}" for v in omegas[i]],
                f"{distances[i]:.4f}",
            ])

    print(f"box_size={box_size}, inertia_ratio={inertia}")
    print(f"process_model={args.process_model}, q0={q0}, omega0={omega0}")
    print(f"distance_profile={args.distance_profile}, far={args.far_distance}, near={args.near_distance}")
    print(f"{len(times)}フレーム ({args.duration}秒 @ {args.fps}fps) を書き出しました: {out_path}")


if __name__ == "__main__":
    main()
