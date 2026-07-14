import torch


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sign-invariant quaternion loss used for silhouette-to-pose learning.

    q and -q represent the same physical orientation, so the absolute dot
    product is used. This is close to the log distance used in SilhoNet.
    """

    pred = normalize_quaternion(pred)
    target = normalize_quaternion(target)
    dots = (pred * target).sum(dim=-1).abs().clamp(max=1.0)
    return torch.log(eps + 1.0 - dots).mean()


def quaternion_angle_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = normalize_quaternion(pred)
    target = normalize_quaternion(target)
    dots = (pred * target).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(dots))


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Quaternion product q ⊗ r, broadcasting over leading dims.

    Convention: (w, x, y, z).  Inputs (..., 4) → output (..., 4).
    """
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rw, rx, ry, rz = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return torch.stack([
        qw*rw - qx*rx - qy*ry - qz*rz,
        qw*rx + qx*rw + qy*rz - qz*ry,
        qw*ry - qx*rz + qy*rw + qz*rx,
        qw*rz + qx*ry - qy*rx + qz*rw,
    ], dim=-1)


# 直方体の対称回転: 各軸 180° 回転 (物体座標系の対称性)
# q_rot(180°, axis) = [0, axis] なので単位ベクトル軸そのものが虚部になる
_BOX_SYMMETRIES = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # identity
    [0.0, 1.0, 0.0, 0.0],  # Rx(180°)
    [0.0, 0.0, 1.0, 0.0],  # Ry(180°)
    [0.0, 0.0, 0.0, 1.0],  # Rz(180°)
], dtype=torch.float32)


def _best_sym_dots(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """4 つの対称等価姿勢との |dot| の最大値を返す。shape: [B]

    各等価姿勢は target ⊗ sym (右から対称回転 = 物体座標系への回転)。
    q と -q は同一回転なので abs() を取る。
    """
    syms = _BOX_SYMMETRIES.to(pred.device)                          # [4, 4]
    # target.unsqueeze(1): [B, 1, 4],  syms.unsqueeze(0): [1, 4, 4]
    # → target_syms[b, i] = target[b] ⊗ syms[i],  shape [B, 4, 4]
    target_syms = quat_mul(target.unsqueeze(1), syms.unsqueeze(0))  # [B, 4, 4]
    dots = (pred.unsqueeze(1) * target_syms).sum(dim=-1).abs().clamp(max=1.0)  # [B, 4]
    return dots.max(dim=1).values                                    # [B]


def symmetry_aware_quaternion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """直方体対称性を考慮した quaternion ロス (スカラー)。

    4 つの等価姿勢のうち最も近いものに対するロスを使う。
    """
    pred = normalize_quaternion(pred)
    target = normalize_quaternion(target)
    return torch.log(eps + 1.0 - _best_sym_dots(pred, target)).mean()


def symmetry_aware_angle_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """直方体対称性を考慮した測地線角度誤差 [deg]。shape: [B]

    4 つの等価姿勢のうち最も近いものとの角度を返す。
    """
    pred = normalize_quaternion(pred)
    target = normalize_quaternion(target)
    return torch.rad2deg(2.0 * torch.acos(_best_sym_dots(pred, target)))
