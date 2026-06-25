import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SilhouettePoseNet(nn.Module):
    """Estimate 3D orientation from a binary silhouette.

    Follows SilhoNet stage-2: ResNet-18 backbone with two FC layers,
    outputting an L2-normalized quaternion.
    Input shape: [B, 1, H, W], typically H=W=64.
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        resnet = models.resnet18(weights=None)

        # Adapt for single-channel binary silhouette (original is 3-channel RGB)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        q = self.head(x)
        return F.normalize(q, p=2, dim=1)
