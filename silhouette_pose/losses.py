import torch


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-4,
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
