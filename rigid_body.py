"""無トルク剛体の回転運動シミュレーション (numpy のみ、torch 不要)。

直方体 (箱) を対象に、クォータニオン運動学とオイラーの剛体運動方程式を積分する。
EKF の予測ステップ (ekf.py) と、合成検証用の軌道生成 (generate_trajectory.py) の
両方から共有される、力学部分のみを担うモジュール (フィルタの概念は含まない)。

状態ベクトル: state = [qw,qx,qy,qz, wx,wy,wz] (7,)
  - q: 物体座標系→カメラ/ワールド座標系への回転を表す単位クォータニオン (w,x,y,z)
  - w: 物体座標系(body frame)で表した角速度 [rad/s]
"""
import math

import numpy as np

from quat_np import quat_mul, quat_normalize


def principal_inertia(box_size: tuple[float, float, float]) -> np.ndarray:
    """一様密度の直方体の主慣性モーメント比 (質量・係数を除いた形) を返す。

    box_size=(w,h,d) の x/y/z 軸まわりの慣性モーメントは、それぞれ
    残り2辺の2乗和に比例する: (h²+d², w²+d², w²+h²)。
    質量が未知でもオイラーの運動方程式は比だけで閉じるため、ここでは
    比のまま (係数 m/12 を省いて) 返す。
    """
    w, h, d = box_size
    return np.array([h*h + d*d, w*w + d*d, w*w + h*h], dtype=np.float64)


def euler_rhs(omega: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """無トルクのオイラーの剛体運動方程式 dω/dt (body frame)。"""
    w1, w2, w3 = omega
    i1, i2, i3 = inertia
    return np.array([
        (i2 - i3) / i1 * w2 * w3,
        (i3 - i1) / i2 * w3 * w1,
        (i1 - i2) / i3 * w1 * w2,
    ], dtype=np.float64)


def state_derivative(state: np.ndarray, inertia: np.ndarray, dynamics: str) -> np.ndarray:
    """状態 [q(4), ω(3)] の時間微分を返す。

    dynamics="constant": dω/dt = 0 (角速度一定モデル)
    dynamics="euler":    dω/dt = euler_rhs (無トルク剛体の歳差/ジャニベコフ運動)
    """
    q = state[:4]
    omega = state[4:]
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float64)
    dq = 0.5 * quat_mul(q, omega_quat)

    if dynamics == "constant":
        domega = np.zeros(3, dtype=np.float64)
    elif dynamics == "euler":
        domega = euler_rhs(omega, inertia)
    else:
        raise ValueError(f"unknown dynamics: {dynamics!r}")

    return np.concatenate([dq, domega])


def rk4_step(state: np.ndarray, dt: float, inertia: np.ndarray, dynamics: str) -> np.ndarray:
    """状態を dt だけ RK4 で1ステップ積分し、クォータニオン部を再正規化して返す。"""
    k1 = state_derivative(state, inertia, dynamics)
    k2 = state_derivative(state + 0.5 * dt * k1, inertia, dynamics)
    k3 = state_derivative(state + 0.5 * dt * k2, inertia, dynamics)
    k4 = state_derivative(state + dt * k3, inertia, dynamics)
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    new_state[:4] = quat_normalize(new_state[:4])
    return new_state


def integrate_trajectory(
    q0: np.ndarray,
    omega0: np.ndarray,
    dt: float,
    n_steps: int,
    inertia: np.ndarray,
    dynamics: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """初期状態から n_steps ステップ積分し、(times, quats, omegas) を返す。

    times: [n_steps+1], quats: [n_steps+1, 4], omegas: [n_steps+1, 3]
    (初期状態 t=0 を含めた n_steps+1 点)
    """
    state = np.concatenate([
        quat_normalize(np.asarray(q0, dtype=np.float64)),
        np.asarray(omega0, dtype=np.float64),
    ])
    times = np.zeros(n_steps + 1, dtype=np.float64)
    quats = np.zeros((n_steps + 1, 4), dtype=np.float64)
    omegas = np.zeros((n_steps + 1, 3), dtype=np.float64)

    quats[0] = state[:4]
    omegas[0] = state[4:]
    for i in range(1, n_steps + 1):
        state = rk4_step(state, dt, inertia, dynamics)
        times[i] = i * dt
        quats[i] = state[:4]
        omegas[i] = state[4:]

    return times, quats, omegas
