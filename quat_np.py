"""クォータニオン演算の numpy 実装 (torch 不要)。

silhouette_pose.losses の torch 実装と同じ規約 (w,x,y,z, Hamilton積, q と -q は
同一回転) を共有する。逐次推定 (EKF) や合成軌道生成など、torch/autograd を
必要としない箇所から使うための共通モジュール。
"""
import math

import numpy as np

# 直方体の対称回転: 各軸 180° 回転 (物体座標系の対称性)
# silhouette_pose.losses._BOX_SYMMETRIES と同一の値。
BOX_SYMMETRIES = np.array([
    [1.0, 0.0, 0.0, 0.0],  # identity
    [0.0, 1.0, 0.0, 0.0],  # Rx(180°)
    [0.0, 0.0, 1.0, 0.0],  # Ry(180°)
    [0.0, 0.0, 0.0, 1.0],  # Rz(180°)
], dtype=np.float32)


def quat_normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, eps)


def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """クォータニオン積 q ⊗ r。(w,x,y,z) 規約, 末尾軸をブロードキャストする。

    入力の dtype をそのまま保持する (float32 に強制すると、rigid_body.py の
    数値微分ヤコビアンのような小さい epsilon を使う計算で精度不足になるため)。
    """
    q = np.asarray(q)
    r = np.asarray(r)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rw, rx, ry, rz = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        qw*rw - qx*rx - qy*ry - qz*rz,
        qw*rx + qx*rw + qy*rz - qz*ry,
        qw*ry - qx*rz + qy*rw + qz*rx,
        qw*rz + qx*ry - qy*rx + qz*rw,
    ], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """単位クォータニオン [w,x,y,z] を回転行列に変換する (v' = R v)。"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z),     2*(x*z+w*y)],
        [2*(x*y+w*z),     1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x),     1 - 2*(x*x+y*y)],
    ], dtype=np.float32)


def nearest_sym_quat(pred_q: np.ndarray, target_q: np.ndarray) -> tuple[np.ndarray, int]:
    """対称性を考慮し、pred に最も近い target の等価姿勢と、採用した対称インデックスを返す。

    各等価姿勢は target ⊗ sym (右から対称回転 = 物体座標系への回転)。
    q と -q は同一回転なので |dot| の絶対値で比較する。
    """
    p = quat_normalize(pred_q)
    t = quat_normalize(target_q)

    best_dot = -1.0
    best_idx = 0
    best_equiv = t
    for i, sym in enumerate(BOX_SYMMETRIES):
        equiv = quat_mul(t, sym)
        d = abs(float(np.dot(p, equiv)))
        if d > best_dot:
            best_dot = d
            best_idx = i
            best_equiv = equiv
    return best_equiv, best_idx


def symmetry_aware_angle_error_deg(pred_q: np.ndarray, target_q: np.ndarray) -> float:
    """直方体対称性を考慮した測地線角度誤差 [deg]。"""
    p = quat_normalize(pred_q)
    equiv, _ = nearest_sym_quat(pred_q, target_q)
    dot = min(1.0, abs(float(np.dot(p, equiv))))
    return math.degrees(2.0 * math.acos(dot))


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """単位クォータニオンを回転ベクトル (log map, 軸×角度 [rad]) に変換する。

    w≈±1 付近 (角度が 0 に近い) で `acos` + 閾値打ち切りを使うと、
    数値微分の中心差分で使うような極小角 (角度 ~ 1e-8 rad 程度) が
    ちょうど閾値未満になり、非ゼロの真値が丸ごとゼロに潰れてしまう
    (EKF の遷移ヤコビアンでこの潰れが起きると、角速度の摂動が姿勢誤差に
    伝わらず学習されないという重大なバグになる)。
    atan2 + sinc を使えば閾値分岐なしに全角度域で数値的に安定する。
    """
    q = quat_normalize(q)
    w = float(q[0])
    xyz = q[1:]
    norm_xyz = float(np.linalg.norm(xyz))
    half = math.atan2(norm_xyz, w)  # [0, pi], acos より数値的に安定
    sinc_half = float(np.sinc(half / math.pi))  # sin(half)/half, half=0 でも 1
    return xyz * (2.0 / sinc_half)


def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    """回転ベクトル (軸×角度 [rad]) を単位クォータニオンに変換する (exp map)。

    np.sinc は正規化 sinc (sin(pi x)/(pi x)) を使うため、
    half=angle/2 として sin(half)/half = sinc(half/pi) と書ける
    (angle=0 でも 0 除算せず sinc(0)=1 として安全に扱える)。
    """
    v = np.asarray(v, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    half = angle / 2.0
    w = math.cos(half)
    sinc_half = float(np.sinc(half / math.pi))
    xyz = 0.5 * v * sinc_half
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)


def sample_uniform_quat(rng: np.random.Generator) -> np.ndarray:
    """SO(3) 上一様なクォータニオンをサンプリングする (Shoemake の方法)。

    matlab/generate_random_database.m のサンプリング式と同一。
    """
    u1, u2, u3 = rng.random(3)
    qw = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    qx = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    qy = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    qz = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return np.array([qw, qx, qy, qz], dtype=np.float32)
