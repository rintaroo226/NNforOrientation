"""rigid_body.py (剛体回転シミュレーション) の動作確認スクリプト (torch 不要)。

実行: python3 verify_rigid_body.py
"""
import math

import numpy as np

from quat_np import BOX_SYMMETRIES, quat_mul
from rigid_body import euler_rhs, integrate_trajectory, principal_inertia

ATOL = 1e-4
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


def main() -> None:
    all_passed = True
    box_size = (3.0, 2.0, 1.0)
    inertia = principal_inertia(box_size)

    print("=== 動作確認: rigid_body.py ===\n")

    # 1. 慣性モーメント比
    print("1. box_size=(3,2,1) の慣性モーメント比が (5,10,13)")
    ok = check("principal_inertia", np.allclose(inertia, [5, 10, 13]), f"inertia={inertia}")
    all_passed &= ok
    print()

    # 2. dynamics="constant" は dω/dt=0、主軸ぴったりの回転は半周期後に180°対称姿勢と一致
    print("2. dynamics=\"constant\": 主軸(x)ぴったりの回転が半周期後に Rx(180°) と一致")
    w = 2.0
    omega0 = np.array([w, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    n_steps = 2000
    dt = (math.pi / w) / n_steps
    times, quats, omegas = integrate_trajectory(q0, omega0, dt, n_steps, inertia, "constant")
    omega_const = np.allclose(omegas, omega0, atol=1e-8)
    q_final = quats[-1]
    matches_sym = any(
        np.allclose(np.abs(q_final), np.abs(BOX_SYMMETRIES[i]), atol=1e-3) for i in range(4)
    )
    ok = check("dω/dt=0 (角速度が変化しない)", omega_const)
    all_passed &= ok
    ok = check("半周期後に対称姿勢と一致", matches_sym, f"q_final={q_final}")
    all_passed &= ok
    norms = np.linalg.norm(quats, axis=1)
    ok = check("クォータニオンの単位ノルムが保たれる", np.allclose(norms, 1.0, atol=1e-6))
    all_passed &= ok
    print()

    # 3. dynamics="euler": エネルギー・角運動量保存
    print("3. dynamics=\"euler\": エネルギー・角運動量 |L|^2 の保存")
    rng = np.random.default_rng(1)
    omega0 = rng.normal(size=3) * 1.5
    dt, n_steps = 0.001, 5000
    times, quats, omegas = integrate_trajectory(q0, omega0, dt, n_steps, inertia, "euler")
    energy = 0.5 * np.sum(inertia * omegas ** 2, axis=1)
    l_mag2 = np.sum((inertia * omegas) ** 2, axis=1)
    energy_drift = (energy.max() - energy.min()) / energy[0]
    l_drift = (l_mag2.max() - l_mag2.min()) / l_mag2[0]
    ok = check("エネルギー保存 (相対ドリフト < 1e-4)", energy_drift < 1e-4, f"drift={energy_drift:.2e}")
    all_passed &= ok
    ok = check("|L|^2 保存 (相対ドリフト < 1e-4)", l_drift < 1e-4, f"drift={l_drift:.2e}")
    all_passed &= ok
    print()

    # 4. ジャニベコフ効果: 中間軸(y, I=10)付近は不安定 → 反転が起きる
    print("4. ジャニベコフ効果: 中間軸(y軸)付近のスピンは不安定 (反転する)")
    dt, n_steps = 0.002, 20000

    def count_flips(omega0: np.ndarray, axis_idx: int) -> int:
        _, _, om = integrate_trajectory(q0, omega0, dt, n_steps, inertia, "euler")
        signs = np.sign(om[:, axis_idx])
        return int(np.sum(np.diff(signs) != 0))

    flips_intermediate = count_flips(np.array([1e-3, 5.0, 1e-3]), axis_idx=1)
    ok = check("中間軸(y)スピンで反転が発生する", flips_intermediate > 0, f"flips={flips_intermediate}")
    all_passed &= ok
    print()

    # 5. 最大軸・最小軸まわりのスピンは安定 (反転しない)
    print("5. 最大軸(z)・最小軸(x)まわりのスピンは安定 (反転しない)")
    flips_major = count_flips(np.array([1e-3, 1e-3, 5.0]), axis_idx=2)
    flips_minor = count_flips(np.array([5.0, 1e-3, 1e-3]), axis_idx=0)
    ok = check("最大軸(z, I=13)スピンで反転しない", flips_major == 0, f"flips={flips_major}")
    all_passed &= ok
    ok = check("最小軸(x, I=5)スピンで反転しない", flips_minor == 0, f"flips={flips_minor}")
    all_passed &= ok
    print()

    # 6. Jacobian のステップ幅ロバスト性 (中心差分の epsilon を変えても近い値になること)
    print("6. state_derivative の中心差分ヤコビアンが epsilon に対してロバスト")
    from rigid_body import state_derivative

    def numeric_jacobian(state: np.ndarray, eps: float) -> np.ndarray:
        n = len(state)
        jac = np.zeros((n, n))
        for i in range(n):
            dstate = np.zeros(n)
            dstate[i] = eps
            f_plus = state_derivative(state + dstate, inertia, "euler")
            f_minus = state_derivative(state - dstate, inertia, "euler")
            jac[:, i] = (f_plus - f_minus) / (2 * eps)
        return jac

    state0 = np.concatenate([q0.astype(np.float64), np.array([0.5, 1.2, -0.3])])
    jac_a = numeric_jacobian(state0, eps=1e-4)
    jac_b = numeric_jacobian(state0, eps=1e-6)
    jac_diff = np.max(np.abs(jac_a - jac_b))
    ok = check("epsilon=1e-4 と 1e-6 のヤコビアンが一致", jac_diff < 1e-3, f"max_diff={jac_diff:.2e}")
    all_passed &= ok
    print()

    print("=" * 40)
    if all_passed:
        print(f"  {PASS} 全テスト通過")
    else:
        print(f"  {FAIL} 失敗あり")
    print("=" * 40)


if __name__ == "__main__":
    main()
