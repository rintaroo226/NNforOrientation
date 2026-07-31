"""直方体の姿勢+角速度を推定する乗法型 EKF (Multiplicative EKF)。

torch 不要 (numpy のみ)。quat_np.py / rigid_body.py の上に構築される。

状態表現:
    公称状態: (q̂ ∈ S³, ω̂ ∈ R³)   ... q̂ は物体座標系→カメラ/ワールド座標系の回転
    誤差状態: δx = (δθ, δω) ∈ R⁶  ... 真の状態は q = q̂ ⊗ δq, δq ≈ [1, δθ/2],
                                      ω = ω̂ + δω

**δθ は物体座標系 (body frame) で定義する** (クォータニオン運動学・オイラーの
運動方程式がどちらも body frame 表現のため。左右反転バグを経験したプロジェクト
であることを踏まえ、この規約を明記する)。

遷移ヤコビアン F は解析的に導出せず、rigid_body.rk4_step を数値微分 (中心差分)
することで得る。6次元状態なので毎回 12 回の rk4_step 呼び出しで済み、実時間性能
上は無視できるコスト。
"""
import math
from dataclasses import dataclass, field

import numpy as np

from quat_np import (
    nearest_sym_quat,
    quat_conjugate,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rotvec_from_quat,
)
from rigid_body import rk4_step


@dataclass
class EKFConfig:
    process_model: str = "constant"       # "constant" | "euler"
    inertia: np.ndarray | None = None      # dynamics="euler" のとき必須 (rigid_body.principal_inertia の出力)
    q_theta_rw: float = 1e-5               # 姿勢誤差のランダムウォーク分散密度 [rad^2/s]
    q_omega_rw: float = 1e-3               # 角速度のランダムウォーク分散密度 [(rad/s)^2/s]
    meas_noise_deg: float = 2.0            # 観測(NN予測)の角度誤差の標準偏差 [deg]
    gate_mode: str = "both"                # "chi2" | "fixed" | "both" | "none"
    gate_chi2_threshold: float = 7.815     # カイ二乗検定のしきい値 (自由度3, 95%)
    gate_max_angle_deg: float = 30.0        # 固定角度上限によるゲーティング [deg]
    jacobian_eps: float = 1e-6


@dataclass
class UpdateResult:
    accepted: bool
    innovation_deg: float
    nis: float
    sym_branch_idx: int


def _retract(q: np.ndarray, omega: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """誤差状態 d=(δθ,δω) を公称状態 (q,omega) に適用する (retraction)。"""
    dtheta, domega = d[:3], d[3:]
    q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dtheta)))
    return q_new, omega + domega


def _error_state(q_nom: np.ndarray, omega_nom: np.ndarray,
                  q_pert: np.ndarray, omega_pert: np.ndarray) -> np.ndarray:
    """公称状態に対する摂動状態の誤差 (δθ,δω) を返す (body frame)。"""
    err_quat = quat_mul(quat_conjugate(q_nom), q_pert)
    if err_quat[0] < 0:
        err_quat = -err_quat  # q と -q は同一回転。短い方の回転を採用する。
    dtheta = rotvec_from_quat(err_quat)
    domega = omega_pert - omega_nom
    return np.concatenate([dtheta, domega])


class BoxOrientationEKF:
    def __init__(
        self,
        q0: np.ndarray,
        omega0: np.ndarray,
        P0: np.ndarray,
        config: EKFConfig,
    ) -> None:
        if config.process_model == "euler" and config.inertia is None:
            raise ValueError("process_model='euler' には config.inertia が必要です")
        self.config = config
        self.q = quat_normalize(np.asarray(q0, dtype=np.float64))
        self.omega = np.asarray(omega0, dtype=np.float64).copy()
        self.P = np.asarray(P0, dtype=np.float64).copy()
        self._inertia = (
            np.asarray(config.inertia, dtype=np.float64)
            if config.inertia is not None
            else np.ones(3)  # dynamics="constant" では未使用
        )

    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.q.copy(), self.omega.copy(), self.P.copy()

    # -------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        state0 = np.concatenate([self.q, self.omega])
        state1 = rk4_step(state0, dt, self._inertia, self.config.process_model)
        q1, omega1 = state1[:4], state1[4:]

        F = self._transition_jacobian(state0, q1, omega1, dt)
        Q = self._process_noise(dt)

        self.q = quat_normalize(q1)
        self.omega = omega1
        self.P = F @ self.P @ F.T + Q

    def _transition_jacobian(
        self, state0: np.ndarray, q1: np.ndarray, omega1: np.ndarray, dt: float,
    ) -> np.ndarray:
        eps = self.config.jacobian_eps
        q0, omega0 = state0[:4], state0[4:]
        F = np.zeros((6, 6))
        for i in range(6):
            d_plus = np.zeros(6)
            d_plus[i] = eps
            q_p, omega_p = _retract(q0, omega0, d_plus)
            next_p = rk4_step(np.concatenate([q_p, omega_p]), dt, self._inertia, self.config.process_model)
            err_p = _error_state(q1, omega1, next_p[:4], next_p[4:])

            d_minus = np.zeros(6)
            d_minus[i] = -eps
            q_m, omega_m = _retract(q0, omega0, d_minus)
            next_m = rk4_step(np.concatenate([q_m, omega_m]), dt, self._inertia, self.config.process_model)
            err_m = _error_state(q1, omega1, next_m[:4], next_m[4:])

            F[:, i] = (err_p - err_m) / (2 * eps)
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * self.config.q_theta_rw * dt
        Q[3:, 3:] = np.eye(3) * self.config.q_omega_rw * dt
        return Q

    # -------------------------------------------------------------------
    # update
    # -------------------------------------------------------------------

    def update(self, q_meas: np.ndarray) -> UpdateResult:
        q_prior, omega_prior, P_prior = self.q, self.omega, self.P

        sym_equiv, sym_idx = nearest_sym_quat(pred_q=q_prior, target_q=q_meas)

        err_quat = quat_mul(quat_conjugate(q_prior), sym_equiv)
        if err_quat[0] < 0:
            err_quat = -err_quat
        y = rotvec_from_quat(err_quat)  # innovation [rad], body frame

        R = np.eye(3) * math.radians(self.config.meas_noise_deg) ** 2
        S = P_prior[:3, :3] + R
        S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)
        innovation_deg = math.degrees(float(np.linalg.norm(y)))

        accepted = self._gate(nis, innovation_deg)

        if accepted:
            K = P_prior[:, :3] @ S_inv  # (6,3), H=[I3,0] より P H^T = P[:, :3]
            dx = K @ y

            q_post = quat_normalize(quat_mul(q_prior, quat_from_rotvec(dx[:3])))
            omega_post = omega_prior + dx[3:]

            KH = np.zeros((6, 6))
            KH[:, :3] = K
            I_KH = np.eye(6) - KH
            P_post = I_KH @ P_prior @ I_KH.T + K @ R @ K.T

            self.q, self.omega, self.P = q_post, omega_post, P_post

        return UpdateResult(
            accepted=accepted,
            innovation_deg=innovation_deg,
            nis=nis,
            sym_branch_idx=sym_idx,
        )

    def _gate(self, nis: float, innovation_deg: float) -> bool:
        mode = self.config.gate_mode
        chi2_ok = nis < self.config.gate_chi2_threshold
        angle_ok = innovation_deg < self.config.gate_max_angle_deg
        if mode == "chi2":
            return chi2_ok
        if mode == "fixed":
            return angle_ok
        if mode == "both":
            return chi2_ok and angle_ok
        if mode == "none":
            return True
        raise ValueError(f"unknown gate_mode: {mode!r}")
