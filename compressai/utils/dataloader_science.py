import math
from typing import Tuple, Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F

NormType = Literal["none", "minmax", "meanstd"]

def normalize_tensor(
    x: torch.Tensor,
    norm_type: NormType = "minmax",
    *,
    data_min: float = 0.0,
    data_max: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    if norm_type == "none":
        print("Suggestion: consider applying normalization for better model performance.")
        print("You can choose 'minmax' or 'meanstd' normalization.")
        return x
    if norm_type == "minmax":
        denom = (data_max - data_min)
        if denom == 0:
            raise ValueError("minmax normalization requires data_max != data_min")
        return (x - data_min) / denom
    if norm_type == "meanstd":
        if std == 0:
            raise ValueError("meanstd normalization requires std != 0")
        return (x - mean) / std
    raise ValueError(f"Unknown norm_type: {norm_type}")


def denormalize_tensor(
    x: torch.Tensor,
    norm_type: NormType = "none",
    *,
    data_min: float = 0.0,
    data_max: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    if norm_type == "none":
        return x
    if norm_type == "minmax":
        return x * (data_max - data_min) + data_min
    if norm_type == "meanstd":
        return x * std + mean
    raise ValueError(f"Unknown norm_type: {norm_type}")

def make_weight_window(bh: int, bw: int, device, dtype):
    # Hann window 
    wh = torch.hann_window(bh, periodic=False, device=device, dtype=dtype)
    ww = torch.hann_window(bw, periodic=False, device=device, dtype=dtype)
    w2d = wh[:, None] * ww[None, :]          # (bh,bw)
    w2d = w2d.clamp_min(1e-6)                # 避免全0
    return w2d


def load_bin_nhw(
    bin_path: str,
    input_shape: Tuple[int, int, int],
    *,
    dtype: np.dtype = np.uint8,
    device: Optional[torch.device] = None,
    norm_type: NormType = "none",
) -> torch.Tensor:
    """
    Step 1: load .bin -> (N,H,W)
    """
    n, h, w = input_shape
    need = n * h * w

    arr = np.fromfile(bin_path, dtype=dtype)
    if arr.size < need:
        raise ValueError(f"Bin too small: need {need} elems for {input_shape}, got {arr.size}")
    if arr.size > need:
        arr = arr[:need]
    arr = arr.reshape(n, h, w)

    x_ori = torch.from_numpy(arr)
    
    data_min = float(arr.min())
    data_max = float(arr.max())
    mean = float(arr.mean())
    std = float(arr.std())

    
    x_norm = normalize_tensor(
        x_ori,
        norm_type,
        data_min=data_min,
        data_max=data_max,
        mean=mean,
        std=std
    )

    if device is not None:
        x_norm = x_norm.to(device)
        x_ori = x_ori.to(device)
    return x_norm, x_ori, {"min": data_min, "max": data_max, "mean": mean, "std": std}

def extract_overlapped_blocks(
    x_g3hw: torch.Tensor,                 # (G,3,H,W)
    block_hw: Tuple[int, int],            # (bh,bw) model input size
    stride_hw: Tuple[int, int],           # (sh,sw) < (bh,bw) for overlap
    *,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Extract overlapped REAL blocks of size (3,bh,bw) using sliding window.

    We do ONE global pad on H/W so every window is valid, then extract.
    Return:
      blocks: (bn,3,bh,bw)
      meta: (H0, W0, Hp, Wp) where:
        - H0,W0 = original H,W
        - Hp,Wp = padded H,W used for window extraction
    """
    if x_g3hw.ndim != 4 or x_g3hw.shape[1] != 3:
        raise ValueError(f"Expected (G,3,H,W), got {tuple(x_g3hw.shape)}")

    bh, bw = block_hw
    sh, sw = stride_hw
    if sh <= 0 or sw <= 0:
        raise ValueError("stride must be positive")
    if sh >= bh or sw >= bw:
        raise ValueError("For overlap, require stride < block size (bh,bw)")

    G, _, H, W = x_g3hw.shape

    # We want start positions: 0, sh, 2sh, ... <= H-1 and ensure window covers end
    # Pad so that last window starting at last_h starts within range and ends at <= Hp
    # Compute how many steps
    n_h = math.ceil((H - bh) / sh) + 1 if H > bh else 1
    n_w = math.ceil((W - bw) / sw) + 1 if W > bw else 1
    Hp = (n_h - 1) * sh + bh
    Wp = (n_w - 1) * sw + bw

    pad_bottom = max(0, Hp - H)
    pad_right = max(0, Wp - W)

    if pad_bottom > 0 or pad_right > 0:
        # pad (left,right,top,bottom) on last two dims
        x_pad = F.pad(x_g3hw, (0, pad_right, 0, pad_bottom), mode="constant", value=pad_value)
    else:
        x_pad = x_g3hw

    # Extract windows
    blocks = []
    for g in range(G):
        for ih in range(n_h):
            h0 = ih * sh
            h1 = h0 + bh
            for iw in range(n_w):
                w0 = iw * sw
                w1 = w0 + bw
                blocks.append(x_pad[g, :, h0:h1, w0:w1])  # (3,bh,bw)

    return torch.stack(blocks, dim=0), (H, W, Hp, Wp)

def blocks_to_nhw_overlap_weighted(
    blocks_bn3bhbw: torch.Tensor,      # (bn,3,bh,bw) 模型输出块
    input_shape: Tuple[int, int, int], # (N,H,W)
    block_size: Tuple[int, int, int],  # (3,bh,bw)
    stride_hw: Tuple[int, int],        # (sh,sw)
    meta_hw: Tuple[int, int, int, int],# (H0,W0,Hp,Wp)
    *,
    pad_value: float = 0.0,
    norm_type: NormType = "minmax",
    data_min: float = 0.0,
    data_max: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    N, H0, W0 = input_shape
    c, bh, bw = block_size
    sh, sw = stride_hw
    H, W, Hp, Wp = meta_hw
    assert (H, W) == (H0, W0)

    # 计算 extraction 时的网格数量（必须和 extract_overlapped_blocks 一致）
    n_h = math.ceil((H - bh) / sh) + 1 if H > bh else 1
    n_w = math.ceil((W - bw) / sw) + 1 if W > bw else 1

    G = math.ceil(N / 3)
    bn_expected = G * n_h * n_w
    if blocks_bn3bhbw.shape[0] != bn_expected:
        raise ValueError(f"bn mismatch: expected {bn_expected}, got {blocks_bn3bhbw.shape[0]}")

    device = blocks_bn3bhbw.device
    dtype = blocks_bn3bhbw.dtype

    # 加权窗（bh,bw）
    w2d = make_weight_window(bh, bw, device=device, dtype=dtype)  # (bh,bw)
    w2d = w2d[None, None, :, :]  # (1,1,bh,bw) 便于broadcast

    # sum/weight 画布：用 Hp,Wp（和你 extraction pad 后一致）
    sum_canvas = torch.zeros((G, 3, Hp, Wp), device=device, dtype=dtype)
    w_canvas   = torch.zeros((G, 1, Hp, Wp), device=device, dtype=dtype)

    idx = 0
    for g in range(G):
        for ih in range(n_h):
            h0 = ih * sh
            h1 = h0 + bh
            for iw in range(n_w):
                w0 = iw * sw
                w1 = w0 + bw

                block = blocks_bn3bhbw[idx]  # (3,bh,bw)
                block = block[None, :, :, :] # (1,3,bh,bw)

                sum_canvas[g:g+1, :, h0:h1, w0:w1] += block * w2d
                w_canvas[g:g+1, :, h0:h1, w0:w1]   += w2d
                idx += 1

    # 归一化除权重（避免除0）
    out = sum_canvas / w_canvas.clamp_min(1e-6)

    # 裁回原图大小并 reshape 回 (N,H,W)
    nhw = out[:, :, :H, :W].contiguous().view(G * 3, H, W)[:N]
    
    nhw = nhw.clamp(0, 1)

    # denorm
    nhw = denormalize_tensor(
        nhw, norm_type,
        data_min=data_min, data_max=data_max,
        mean=mean, std=std,
    )
    return nhw

def nhw_to_g3hw(x_nhw: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """
    Convert (N,H,W) -> (G,3,H,W) by grouping N in consecutive 3.
    If N not divisible by 3, pad along N (post-pad) so every group has 3 slices.
    """
    if x_nhw.ndim != 3:
        raise ValueError(f"Expected (N,H,W), got {tuple(x_nhw.shape)}")
    n, h, w = x_nhw.shape
    g = math.ceil(n / 3)
    n2 = g * 3

    if n2 != n:
        out = torch.full((n2, h, w), pad_value, dtype=x_nhw.dtype, device=x_nhw.device)
        out[:n] = x_nhw
        x_nhw = out

    return x_nhw.view(g, 3, h, w)  # (G,3,H,W)


def center_pad_2d(
    x: torch.Tensor,  # (..., H, W)
    target_hw: Tuple[int, int],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Center (balanced) pad last two dims to target (bh,bw).
    """
    bh, bw = target_hw
    H, W = x.shape[-2], x.shape[-1]
    if H > bh or W > bw:
        raise ValueError(f"Cannot pad: current {(H,W)} larger than target {(bh,bw)}")

    pad_h = bh - H
    pad_w = bw - W
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(x, (left, right, top, bottom), mode="constant", value=pad_value)


def cut_then_pad_blocks(
    x_g3hw: torch.Tensor,                 # (G,3,H,W)
    block_hw: Tuple[int, int],            # (bh,bw) final
    p: float,
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Step 2 (as you requested): FIRST cut blocks, THEN pad each block individually.

    - real block size: (3, rh, rw) where rh=ceil(bh*p), rw=ceil(bw*p)
    - cut along H/W into tiles of size rh/rw.
      IMPORTANT: we do NOT pad the whole image to rh/rw multiples first.
      Instead, each edge tile may be smaller (<rh or <rw).
    - each cut tile is then center-padded to (3,bh,bw).

    Output:
      blocks: (bn,3,bh,bw) where bn = G * tiles_h * tiles_w
    """
    if not (0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")
    if x_g3hw.ndim != 4 or x_g3hw.shape[1] != 3:
        raise ValueError(f"Expected (G,3,H,W), got {tuple(x_g3hw.shape)}")

    bh, bw = block_hw
    G, C, H, W = x_g3hw.shape

    rh = max(1, int(math.ceil(bh * p)))
    rw = max(1, int(math.ceil(bw * p)))
    if rh > bh or rw > bw:
        raise ValueError(f"Invalid p: real (rh,rw)=({rh},{rw}) exceeds (bh,bw)=({bh},{bw})")

    tiles_h = math.ceil(H / rh)
    tiles_w = math.ceil(W / rw)

    blocks = []
    for g in range(G):
        for th in range(tiles_h):
            h0 = th * rh
            h1 = min(h0 + rh, H)  # may be smaller on bottom edge
            for tw in range(tiles_w):
                w0 = tw * rw
                w1 = min(w0 + rw, W)  # may be smaller on right edge

                tile = x_g3hw[g, :, h0:h1, w0:w1]         # (3, h', w')  h'<=rh, w'<=rw
                tile = center_pad_2d(tile, (bh, bw), pad_value=pad_value)  # (3,bh,bw)
                blocks.append(tile)

    return torch.stack(blocks, dim=0)  # (bn,3,bh,bw)

def blockify_bin_overlap(
    bin_path: str,
    input_shape: Tuple[int, int, int],
    block_size: Tuple[int, int, int],      # (3,bh,bw)
    p: float,
    stride_hw: Tuple[int, int],            # (sh,sw)
    *,
    dtype: np.dtype = np.uint8,
    pad_value: float = 0.0,
    norm_type: NormType = "minmax",
    device: Optional[torch.device] = None,
):
    c, bh, bw = block_size
    if c != 3:
        raise ValueError("block_size must be (3,bh,bw)")

    x_nhw_norm, x_nhw_ori, status = load_bin_nhw(
        bin_path,
        input_shape,
        dtype=dtype,
        device=device,
        norm_type=norm_type,
    )
    x_g3hw = nhw_to_g3hw(x_nhw_norm, pad_value=pad_value)

    blocks, meta_hw = extract_overlapped_blocks(
        x_g3hw,
        block_hw=(bh, bw),
        stride_hw=stride_hw,
        pad_value=pad_value,
    )
    return blocks, x_nhw_ori, status, meta_hw

def blockify_bin(
    bin_path: str,
    input_shape: Tuple[int, int, int],
    block_size: Tuple[int, int, int],      # (3,bh,bw)
    p: float,
    *,
    dtype: np.dtype = np.uint8,
    pad_value: float = 0.0,
    norm_type: NormType = "minmax",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Step 1: load bin to (N,H,W)
    Step 2: cut blocks (3,rh,rw) first, THEN pad each block to (3,bh,bw)
    Step 3: stack -> (bn,3,bh,bw)
    """
    c, bh, bw = block_size
    if c != 3:
        raise ValueError("block_size must be (3,bh,bw)")

    x_nhw_norm, x_nhw_ori, status = load_bin_nhw(
        bin_path,
        input_shape,
        dtype=dtype,
        device=device,
        norm_type=norm_type,
    )
    x_g3hw = nhw_to_g3hw(x_nhw_norm, pad_value=pad_value)  # (G,3,H,W)
    blocks = cut_then_pad_blocks(x_g3hw, (bh, bw), p, pad_value=pad_value)
    return blocks, x_nhw_ori, status

def center_crop_2d(x: torch.Tensor, crop_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Take centered crop from last two dims.
    x: (..., H, W)
    returns: (..., ch, cw)
    """
    ch, cw = crop_hw
    H, W = x.shape[-2], x.shape[-1]
    if ch > H or cw > W:
        raise ValueError(f"Cannot crop: crop {(ch, cw)} larger than input {(H, W)}")

    top = (H - ch) // 2
    left = (W - cw) // 2
    return x[..., top:top + ch, left:left + cw]


def blocks_to_nhw(
    blocks_bn3bhbw: torch.Tensor,      # (bn,3,bh,bw)
    input_shape: Tuple[int, int, int], # original (N,H,W)
    block_size: Tuple[int, int, int],  # (3,bh,bw)
    p: float,
    *,
    pad_value: float = 0.0,
    norm_type: NormType = "minmax",
    data_min: float = 0.0,
    data_max: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    if not (0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")

    N, H, W = input_shape
    c, bh, bw = block_size
    if c != 3:
        raise ValueError("block_size must be (3,bh,bw)")
    if blocks_bn3bhbw.ndim != 4 or blocks_bn3bhbw.shape[1] != 3:
        raise ValueError(f"Expected blocks (bn,3,bh,bw), got {tuple(blocks_bn3bhbw.shape)}")
    if blocks_bn3bhbw.shape[2] != bh or blocks_bn3bhbw.shape[3] != bw:
        raise ValueError("Block spatial size mismatch")

    rh = max(1, int(math.ceil(bh * p)))
    rw = max(1, int(math.ceil(bw * p)))

    G = math.ceil(N / 3)
    tiles_h = math.ceil(H / rh)
    tiles_w = math.ceil(W / rw)
    bn_expected = G * tiles_h * tiles_w
    if blocks_bn3bhbw.shape[0] != bn_expected:
        raise ValueError(f"bn mismatch: expected {bn_expected}, got {blocks_bn3bhbw.shape[0]}")

    device = blocks_bn3bhbw.device
    dtype = blocks_bn3bhbw.dtype

    canvas = torch.full((G, 3, H, W), pad_value, dtype=dtype, device=device)

    idx = 0
    for g in range(G):
        for th in range(tiles_h):
            h0 = th * rh
            h1 = min(h0 + rh, H)
            h_len = h1 - h0              # <--真实 tile 高度
            for tw in range(tiles_w):
                w0 = tw * rw
                w1 = min(w0 + rw, W)
                w_len = w1 - w0          # <--真实 tile 宽度

                block = blocks_bn3bhbw[idx]  # (3,bh,bw)

                # 关键：按真实尺寸 (h_len,w_len) 从中心裁剪回来
                small = center_crop_2d(block, (h_len, w_len))  # (3,h_len,w_len)

                canvas[g, :, h0:h1, w0:w1] = small
                idx += 1

    nhw_pad = canvas.view(G * 3, H, W)[:N]
    
    nhw_pad = denormalize_tensor(
        nhw_pad,
        norm_type,
        data_min=data_min,
        data_max=data_max,
        mean=mean,
        std=std,
    )
    return nhw_pad


# -------------------- Example --------------------
if __name__ == "__main__":
    bin_path = "your.bin"
    input_shape = (120, 512, 512)      # (N,H,W) -> grouped into (40,3,512,512)
    block_size = (3, 128, 128)
    p = 0.8

    out = blockify_bin(
        bin_path,
        input_shape,
        block_size,
        p,
        dtype=np.float32,
        pad_value=0.0,
        device=None,
    )
    print(out.shape)  # (bn,3,128,128)
