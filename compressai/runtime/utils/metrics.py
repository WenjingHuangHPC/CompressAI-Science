# compressai/runtime/utils/metrics.py
from __future__ import annotations
import torch

@torch.no_grad()
def basic_metrics(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-12):
    x_f = x.float()
    y_f = x_hat.float()

    diff = y_f - x_f
    mse = torch.mean(diff * diff)
    rmse = torch.sqrt(mse)

    maxe = torch.max(torch.abs(diff))

    range_ = torch.max(x_f) - torch.min(x_f)
    nrmse = rmse / range_

    psnr = 20.0 * torch.log10(range_ / (2.0 * rmse + eps))
    return {
        "rmse": float(rmse.item()),
        "nrmse": float(nrmse.item()),
        "maxe": float(maxe.item()),
        "psnr": float(psnr.item()),
    }