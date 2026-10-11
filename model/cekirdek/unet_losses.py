"""
model/cekirdek/unet_losses.py -- segmentasyon egitiminde kullanilan kayip
fonksiyonlari.

Servis edilen model Focal Tversky Loss ile egitildi (alpha=0.7, beta=0.3,
gamma=0.75 -- kucuk/dengesiz lezyon alanlarini BCE+Dice'tan daha agir
cezalandirir). BCEDiceLoss karsilastirma icin denenen alternatif koldur.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p = p.reshape(p.size(0), -1)
        g = target.reshape(target.size(0), -1).float()
        tp = (p * g).sum(dim=1)
        fp = (p * (1 - g)).sum(dim=1)
        fn = ((1 - p) * g).sum(dim=1)
        tversky = (tp + EPS) / (tp + self.alpha * fp + self.beta * fn + EPS)
        return ((1.0 - tversky) ** self.gamma).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce_loss = self.bce(logits, target)
        p = torch.sigmoid(logits).reshape(logits.size(0), -1)
        g = target.reshape(target.size(0), -1)
        inter = (p * g).sum(dim=1)
        soft_dice = (2 * inter + EPS) / (p.sum(dim=1) + g.sum(dim=1) + EPS)
        dice_loss = (1.0 - soft_dice).mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def soft_dice_score(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Egitim/val izleme icin (kayip degil) -- sigmoid + hard 0.5 esik Dice, ornek basina."""
    p = (torch.sigmoid(logits) > 0.5).float().reshape(logits.size(0), -1)
    g = target.float().reshape(target.size(0), -1)
    inter = (p * g).sum(dim=1)
    s = p.sum(dim=1) + g.sum(dim=1)
    dice = torch.where(s > 0, 2 * inter / s, torch.ones_like(s))
    return dice
