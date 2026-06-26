"""torch なしで動く symmetry-aware loss の数学的動作確認スクリプト。

実行: python3 verify_losses.py
"""
import math
import random

# ---------------------------------------------------------------------------
# 純粋 Python によるクォータニオン演算
# ---------------------------------------------------------------------------

Quat = tuple[float, float, float, float]  # (w, x, y, z)


def norm(q: Quat) -> float:
    return math.sqrt(sum(v * v for v in q))


def normalize(q: Quat) -> Quat:
    n = norm(q)
    return tuple(v / n for v in q)  # type: ignore[return-value]


def quat_mul(q: Quat, r: Quat) -> Quat:
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return (
        qw*rw - qx*rx - qy*ry - qz*rz,
        qw*rx + qx*rw + qy*rz - qz*ry,
        qw*ry - qx*rz + qy*rw + qz*rx,
        qw*rz + qx*ry - qy*rx + qz*rw,
    )


def dot(q: Quat, r: Quat) -> float:
    return sum(a * b for a, b in zip(q, r))


def angle_error_deg(pred: Quat, target: Quat) -> float:
    d = min(abs(dot(normalize(pred), normalize(target))), 1.0)
    return math.degrees(2.0 * math.acos(d))


BOX_SYMMETRIES: list[Quat] = [
    (1., 0., 0., 0.),  # identity
    (0., 1., 0., 0.),  # Rx(180°)
    (0., 0., 1., 0.),  # Ry(180°)
    (0., 0., 0., 1.),  # Rz(180°)
]


def symmetry_aware_angle_error_deg(pred: Quat, target: Quat) -> float:
    """4 つの対称等価姿勢との角度誤差の最小値。"""
    p = normalize(pred)
    t = normalize(target)
    best = max(
        abs(dot(p, normalize(quat_mul(t, sym))))
        for sym in BOX_SYMMETRIES
    )
    best = min(best, 1.0)
    return math.degrees(2.0 * math.acos(best))


def rand_unit_quat(rng: random.Random) -> Quat:
    while True:
        q = tuple(rng.gauss(0, 1) for _ in range(4))
        n = norm(q)  # type: ignore[arg-type]
        if n > 1e-8:
            return normalize(q)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# テストユーティリティ
# ---------------------------------------------------------------------------

ATOL = 1e-5
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


# ---------------------------------------------------------------------------
# 検証ケース
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(42)
    quats = [rand_unit_quat(rng) for _ in range(20)]
    all_passed = True

    print("=== 動作確認: symmetry-aware angle error ===\n")

    # 1. pred == target → 0°
    print("1. pred == target のとき 0°")
    for i, q in enumerate(quats):
        err = symmetry_aware_angle_error_deg(q, q)
        ok = err < ATOL
        if not ok:
            all_passed = False
            check(f"  sample {i}", ok, f"angle={err:.6f}°")
    check("pred == target → 0°", all_passed)

    print()

    # 2. pred == -target → 0°  (q と -q は同一回転)
    print("2. pred == -target のとき 0°")
    ok2 = True
    for q in quats:
        neg_q = tuple(-v for v in q)
        err = symmetry_aware_angle_error_deg(neg_q, q)  # type: ignore[arg-type]
        if err >= ATOL:
            ok2 = False
            all_passed = False
            print(f"    angle={err:.6f}° for q={q}")
    check("pred == -target → 0°", ok2)

    print()

    # 3-4. pred == target ⊗ Rx/Ry/Rz(180°) のとき symmetry-aware は 0°、通常は 0° でない
    sym_labels = ["Rx(180°)", "Ry(180°)", "Rz(180°)"]
    for idx, label in enumerate(sym_labels, start=1):
        sym = BOX_SYMMETRIES[idx]
        print(f"3-4. pred == target ⊗ {label}")
        ok_sym = True
        ok_normal_nonzero = True
        for q in quats:
            pred_equiv = normalize(quat_mul(q, sym))
            sym_err   = symmetry_aware_angle_error_deg(pred_equiv, q)
            normal_err = angle_error_deg(pred_equiv, q)
            if sym_err >= ATOL:
                ok_sym = False
                all_passed = False
                print(f"    sym_aware={sym_err:.6f}° (expected 0°)")
            if normal_err < 1.0:
                ok_normal_nonzero = False
                print(f"    normal={normal_err:.2f}° (expected non-zero, symmetry test invalid)")
        check(f"symmetry-aware → 0° (pred == target ⊗ {label})", ok_sym)
        check(f"通常 angle error は 0° でない ({label})", ok_normal_nonzero)
        print()

    # Klein 四元群の閉包確認
    print("5. Klein 四元群の閉包: syms[i] ⊗ syms[j] ∈ 対称群")
    ok_group = True
    for i, si in enumerate(BOX_SYMMETRIES):
        for j, sj in enumerate(BOX_SYMMETRIES):
            product = normalize(quat_mul(si, sj))
            in_group = any(
                all(abs(product[k] - sym[k]) < ATOL or abs(product[k] + sym[k]) < ATOL
                    for k in range(4))
                for sym in BOX_SYMMETRIES
            )
            if not in_group:
                ok_group = False
                all_passed = False
                print(f"    syms[{i}] ⊗ syms[{j}] = {product} は群外")
    check("Klein 四元群閉包", ok_group)

    print()
    print("=" * 40)
    if all_passed:
        print(f"  {PASS} 全テスト通過")
    else:
        print(f"  {FAIL} 失敗あり")
    print("=" * 40)


if __name__ == "__main__":
    main()
