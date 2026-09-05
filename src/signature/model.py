"""
Siamese Network for offline signature verification.

Architecture:
  - Shared encoder: EfficientNet-B0 from timm (pretrained on ImageNet)
  - Projection head: Linear(feat_dim → embed_dim) → ReLU → Linear(embed_dim → embed_dim)
  - Distance metric: cosine distance between the two embeddings

References:
  - HTCSigNet (Pattern Recognition, 2025): Hybrid Transformer-Conv signature network
  - Multi-Scale CNN-CrossViT (Complex & Intelligent Systems, 2025): 98.85% on CEDAR
  - timm: https://github.com/huggingface/pytorch-image-models
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError("Install timm: pip install timm>=1.0.0") from e


class SiameseNet(nn.Module):
    """Siamese network with a shared timm backbone encoder.

    Args:
        backbone: Any timm model name. Defaults to ``efficientnet_b0``.
        embed_dim: Dimensionality of the final embedding vector.
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        embed_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim: int = self.encoder.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        projected = self.proj[0](features)
        batch_norm = self.proj[1]
        if self.training and projected.shape[0] == 1:
            projected = F.batch_norm(
                projected,
                batch_norm.running_mean,
                batch_norm.running_var,
                batch_norm.weight,
                batch_norm.bias,
                training=False,
                eps=batch_norm.eps,
            )
        else:
            projected = batch_norm(projected)
        projected = self.proj[2](projected)
        projected = self.proj[3](projected)
        projected = self.proj[4](projected)
        return F.normalize(projected, dim=1)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return L2-normalised embeddings for both inputs."""
        return self.forward_once(x1), self.forward_once(x2)

    @torch.no_grad()
    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Cosine similarity in [−1, 1]. Higher → more similar."""
        was_training = self.training
        self.eval()
        try:
            e1, e2 = self.forward(x1, x2)
            return F.cosine_similarity(e1, e2)
        finally:
            self.train(was_training)
