# MIT-licensed adaptation: NumPy/Numba/Scipy removed; torch+PIL only.

from enum import Enum
from typing import Optional, Tuple, Union

from PIL import Image
import torch
import torch.nn.functional as F

DROP_MASK_ENERGY = 1e5
KEEP_MASK_ENERGY = 1e3


class OrderMode(str, Enum):
    WIDTH_FIRST = "width-first"
    HEIGHT_FIRST = "height-first"


class EnergyMode(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


TensorOrPIL = Union[torch.Tensor, Image.Image]


# -------------- Utilities: I/O and shapes --------------

def _to_tensor(img: TensorOrPIL) -> torch.Tensor:
    """
    Convert PIL or torch.Tensor -> torch.Tensor (H, W, C), float32 in [0,1].
    Accepts:
      - PIL RGB/RGBA/L
      - torch tensor in HWC or CHW (we normalize to HWC)
    """
    if isinstance(img, torch.Tensor):
        t = img
        if t.dim() == 4 and t.shape[0] == 1:  # (1,H,W,C) or (1,C,H,W)
            t = t.squeeze(0)
        if t.dim() == 3 and t.shape[0] in (1, 3, 4):  # CHW
            t = t.permute(1, 2, 0).contiguous()  # HWC
        t = t.float()
        # Assume already in [0,1] if max <= 1.5; otherwise scale from [0,255]
        if t.max() > 1.5:
            t = t / 255.0
        return t.clamp(0, 1)

    # PIL path
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    bands = len(img.getbands())
    byte = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
    t = byte.view(img.size[1], img.size[0], bands).float() / 255.0  # H, W, C
    return t


def _to_pil(t: torch.Tensor) -> Image.Image:
    """
    Torch tensor (H,W,C) in [0,1] -> PIL Image (RGB/RGBA/L)
    """
    t = t.clamp(0, 1)
    H, W = t.shape[:2]
    C = 1 if t.dim() == 2 else t.shape[2]
    if C == 1:
        mode = "L"
        data = (t.squeeze(-1) * 255.0).byte().contiguous()
        return Image.frombytes(mode, (W, H), data.numpy().tobytes())
    elif C == 3:
        mode = "RGB"
    elif C == 4:
        mode = "RGBA"
    else:
        raise ValueError(f"Unsupported channel count: {C}")
    data = (t * 255.0).byte().contiguous()
    return Image.frombytes(mode, (W, H), data.numpy().tobytes())


def _rgb2gray(rgb: torch.Tensor) -> torch.Tensor:
    """HWC float32 [0,1] -> HW grayscale float32"""
    if rgb.dim() == 2:
        return rgb
    if rgb.shape[2] == 1:
        return rgb.squeeze(-1)
    coeffs = torch.tensor([0.2125, 0.7154, 0.0721], dtype=rgb.dtype, device=rgb.device)
    return (rgb[..., :3] * coeffs).sum(dim=-1)


def _list_enum(enum_class) -> Tuple:
    return tuple(x.value for x in enum_class)


# -------------- Sobel energy (pure torch) --------------

def _sobel(gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sobel gradients via conv2d. gray: (H,W) float32."""
    g = gray.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return gx[0, 0], gy[0, 0]


def _get_energy(gray: torch.Tensor) -> torch.Tensor:
    """Backward energy: |Sobel x| + |Sobel y|, returns (H,W)."""
    assert gray.dim() == 2
    grad_x, grad_y = _sobel(gray)
    return grad_x.abs() + grad_y.abs()


# -------------- Seam helpers (torch only) --------------

def _get_seam_mask(width: int, seam: torch.Tensor) -> torch.Tensor:
    """
    seam: (H,) integer column indices.
    Returns mask (H,W) where True marks pixels to remove (the seam).
    """
    eye = torch.eye(width, dtype=torch.bool, device=seam.device)
    return eye.index_select(0, seam)  # (H,W)


def _remove_seam_mask(src: torch.Tensor, seam_mask: torch.Tensor) -> torch.Tensor:
    """
    Remove one vertical seam from src using a boolean mask (H,W) True at seam columns.
    src: (H,W) or (H,W,C). Returns with width-1.
    """
    H, W = seam_mask.shape
    if src.dim() == 2:
        out = torch.empty((H, W - 1), dtype=src.dtype, device=src.device)
        for r in range(H):
            keep = ~seam_mask[r]
            out[r] = src[r, keep]
        return out
    else:
        C = src.shape[2]
        out = torch.empty((H, W - 1, C), dtype=src.dtype, device=src.device)
        for r in range(H):
            keep = ~seam_mask[r]
            out[r] = src[r, keep, :]
        return out


# -------------- Backward seams (dynamic programming) --------------

def _get_backward_seam(energy: torch.Tensor) -> torch.Tensor:
    """
    energy: (H,W) float32
    returns seam: (H,) int32 with column indices.
    """
    H, W = energy.shape
    device = energy.device
    inf = torch.tensor(float("inf"), dtype=energy.dtype, device=device).view(1)
    cost = torch.cat([inf, energy[0], inf], dim=0)  # (W+2,)
    parent = torch.empty((H, W), dtype=torch.int32, device=device)
    base_idx = torch.arange(-1, W - 1, dtype=torch.int32, device=device)  # (-1..W-2)

    for r in range(1, H):
        choices = torch.stack([cost[:-2], cost[1:-1], cost[2:]], dim=0)  # (3,W)
        min_idx = torch.argmin(choices, dim=0).to(torch.int32) + base_idx  # (W,)
        parent[r] = min_idx
        cost_mid = cost[1:-1]  # (W,)
        # cost_mid = cost_mid[min_idx]? We need cost[1:-1][min_idx] elementwise
        gather = torch.gather(cost_mid, 0, min_idx.to(torch.int64))
        cost[1:-1] = gather + energy[r]

    c = torch.argmin(cost[1:-1]).to(torch.int32)
    seam = torch.empty(H, dtype=torch.int32, device=device)
    for r in range(H - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]
    return seam


def _get_backward_seams(
    gray: torch.Tensor, num_seams: int, aux_energy: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    gray: (H,W) float32
    returns seams mask: (H,W) bool, True where a pixel belongs to one of the N seams (in original coords).
    """
    H, W = gray.shape
    device = gray.device
    seams = torch.zeros((H, W), dtype=torch.bool, device=device)
    rows = torch.arange(H, dtype=torch.int64, device=device)
    idx_map = torch.arange(W, dtype=torch.int64, device=device).expand(H, W).clone()

    energy = _get_energy(gray)
    if aux_energy is not None:
        energy = energy + aux_energy

    for _ in range(num_seams):
        seam = _get_backward_seam(energy).to(torch.int64)
        seams[rows, idx_map[rows, seam]] = True

        seam_mask = _get_seam_mask(gray.shape[1], seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if aux_energy is not None:
            aux_energy = _remove_seam_mask(aux_energy, seam_mask)

        # Recompute energy only near seam
        _, cur_w = energy.shape
        lo = max(0, int(seam.min().item()) - 1)
        hi = min(cur_w, int(seam.max().item()) + 1)
        pad_lo = 1 if lo > 0 else 0
        pad_hi = 1 if hi < cur_w - 1 else 0
        mid_block = gray[:, lo - pad_lo : hi + pad_hi]
        mid_energy = _get_energy(mid_block)[:, pad_lo : mid_block.shape[1] - pad_hi]
        if aux_energy is not None:
            mid_energy = mid_energy + aux_energy[:, lo:hi]
        energy = torch.cat([energy[:, :lo], mid_energy, energy[:, hi + 1 :]], dim=1)

    return seams


# -------------- Forward seams (torch) --------------

def _get_forward_seam(gray: torch.Tensor, aux_energy: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Forward energy dynamic programming.
    gray: (H,W) float32
    """
    H, W = gray.shape
    device = gray.device
    # pad columns on both sides
    g = torch.cat([gray[:, :1], gray, gray[:, -1:]], dim=1)  # (H,W+2)

    inf = torch.tensor(float("inf"), dtype=gray.dtype, device=device).view(1)
    dp = torch.cat([inf, (g[0, 2:] - g[0, :-2]).abs(), inf], dim=0)  # (W+2,)
    parent = torch.empty((H, W), dtype=torch.int32, device=device)
    base_idx = torch.arange(-1, W - 1, dtype=torch.int32, device=device)

    for r in range(1, H):
        curr_shl = g[r, 2:]   # shift left
        curr_shr = g[r, :-2]  # shift right
        cost_mid = (curr_shl - curr_shr).abs()
        if aux_energy is not None:
            cost_mid = cost_mid + aux_energy[r]

        prev_mid = g[r - 1, 1:-1]
        cost_left = cost_mid + (prev_mid - curr_shr).abs()
        cost_right = cost_mid + (prev_mid - curr_shl).abs()

        dp_mid = dp[1:-1]
        dp_left = dp[:-2]
        dp_right = dp[2:]

        choices = torch.stack([cost_left + dp_left, cost_mid + dp_mid, cost_right + dp_right], dim=0)  # (3,W)
        min_idx = torch.argmin(choices, dim=0)  # (W,)
        parent[r] = (min_idx + base_idx).to(torch.int32)

        # dp_mid[:] = min across rows for each column
        # emulate in-place per column to keep graph simple
        for j in range(W):
            dp_mid[j] = choices[min_idx[j], j]

    c = torch.argmin(dp[1:-1]).to(torch.int32)
    seam = torch.empty(H, dtype=torch.int32, device=device)
    for r in range(H - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]
    return seam


def _get_forward_seams(
    gray: torch.Tensor, num_seams: int, aux_energy: Optional[torch.Tensor]
) -> torch.Tensor:
    H, W = gray.shape
    device = gray.device
    seams = torch.zeros((H, W), dtype=torch.bool, device=device)
    rows = torch.arange(H, dtype=torch.int64, device=device)
    idx_map = torch.arange(W, dtype=torch.int64, device=device).expand(H, W).clone()
    for _ in range(num_seams):
        seam = _get_forward_seam(gray, aux_energy).to(torch.int64)
        seams[rows, idx_map[rows, seam]] = True
        seam_mask = _get_seam_mask(gray.shape[1], seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if aux_energy is not None:
            aux_energy = _remove_seam_mask(aux_energy, seam_mask)
    return seams


def _get_seams(
    gray: torch.Tensor, num_seams: int, energy_mode: str, aux_energy: Optional[torch.Tensor]
) -> torch.Tensor:
    gray = gray.to(torch.float32)
    if energy_mode == EnergyMode.BACKWARD:
        return _get_backward_seams(gray, num_seams, aux_energy)
    elif energy_mode == EnergyMode.FORWARD:
        return _get_forward_seams(gray, num_seams, aux_energy)
    else:
        raise ValueError(f"expect energy_mode to be one of {_list_enum(EnergyMode)}, got {energy_mode}")


# -------------- Reduce / Expand width (insert seams) --------------

def _insert_seams(src: torch.Tensor, seams: torch.Tensor, delta_width: int) -> torch.Tensor:
    """
    Insert multiple seams into src (H,W[,C]) using seams mask (H,W) where True marks insertion.
    This is a pure-Python loop per row; fast enough for moderate sizes.
    """
    H, W = seams.shape
    if src.dim() == 2:
        src = src.unsqueeze(-1)
    C = src.shape[2]
    dst = torch.empty((H, W + delta_width, C), dtype=src.dtype, device=src.device)
    for r in range(H):
        dst_col = 0
        for c in range(W):
            if seams[r, c]:
                left = src[r, max(c - 1, 0)]
                right = src[r, c]
                dst[r, dst_col] = (left + right) * 0.5
                dst_col += 1
            dst[r, dst_col] = src[r, c]
            dst_col += 1
    return dst.squeeze(-1) if C == 1 else dst


def _reduce_width(
    src: torch.Tensor,
    delta_width: int,
    energy_mode: str,
    aux_energy: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert src.dim() in (2, 3) and delta_width >= 0
    if src.dim() == 2:
        gray = src
    else:
        gray = _rgb2gray(src)
    to_keep = ~_get_seams(gray, delta_width, energy_mode, aux_energy)
    # Remove columns per row using mask
    if src.dim() == 2:
        H, W = src.shape
        dst = torch.empty((H, W - delta_width), dtype=src.dtype, device=src.device)
        for r in range(H):
            dst[r] = src[r, to_keep[r]]
    else:
        H, W, C = src.shape
        dst = torch.empty((H, W - delta_width, C), dtype=src.dtype, device=src.device)
        for r in range(H):
            dst[r] = src[r, to_keep[r], :]
    if aux_energy is not None:
        H, W = aux_energy.shape
        aux_dst = torch.empty((H, W - delta_width), dtype=aux_energy.dtype, device=src.device)
        for r in range(H):
            aux_dst[r] = aux_energy[r, to_keep[r]]
        aux_energy = aux_dst
    return dst, aux_energy


def _expand_width(
    src: torch.Tensor,
    delta_width: int,
    energy_mode: str,
    aux_energy: Optional[torch.Tensor],
    step_ratio: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert src.dim() in (2, 3) and delta_width >= 0
    if not (0 < step_ratio <= 1):
        raise ValueError(f"expect `step_ratio` in (0,1], got {step_ratio}")

    dst = src
    while delta_width > 0:
        max_step_size = max(1, round(step_ratio * dst.shape[1]))
        step_size = min(max_step_size, delta_width)
        gray = dst if dst.dim() == 2 else _rgb2gray(dst)
        seams = _get_seams(gray, step_size, energy_mode, aux_energy)
        dst = _insert_seams(dst, seams, step_size)
        if aux_energy is not None:
            aux_energy = _insert_seams(aux_energy, seams, step_size)
        delta_width -= step_size

    return dst, aux_energy


def _resize_width(
    src: torch.Tensor,
    width: int,
    energy_mode: str,
    aux_energy: Optional[torch.Tensor],
    step_ratio: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert src.numel() > 0 and src.dim() in (2, 3)
    assert width > 0
    src_w = src.shape[1]
    if src_w < width:
        return _expand_width(src, width - src_w, energy_mode, aux_energy, step_ratio)
    else:
        return _reduce_width(src, src_w - width, energy_mode, aux_energy)


def _transpose_image(src: torch.Tensor) -> torch.Tensor:
    if src.dim() == 3:
        return src.permute(1, 0, 2).contiguous()  # (W,H,C)
    else:
        return src.t().contiguous()


def _resize_height(
    src: torch.Tensor,
    height: int,
    energy_mode: str,
    aux_energy: Optional[torch.Tensor],
    step_ratio: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert src.dim() in (2, 3) and height > 0
    if aux_energy is not None:
        aux_energy = aux_energy.t().contiguous()
    src = _transpose_image(src)
    src, aux_energy = _resize_width(src, height, energy_mode, aux_energy, step_ratio)
    src = _transpose_image(src)
    if aux_energy is not None:
        aux_energy = aux_energy.t().contiguous()
    return src, aux_energy


def _check_mask(mask: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    if isinstance(mask, Image.Image):
        mask = _to_tensor(mask)
    if isinstance(mask, torch.Tensor) and mask.dim() == 3:
        mask = _rgb2gray(mask)
    mask = (mask > 0.5)
    if mask.dim() != 2:
        raise ValueError(f"expect mask 2D, got {tuple(mask.shape)}")
    if tuple(mask.shape) != shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != image shape {shape}")
    return mask


def _check_src(src: TensorOrPIL) -> torch.Tensor:
    t = _to_tensor(src)
    if t.numel() == 0 or t.dim() not in (2, 3):
        raise ValueError(f"expect 3D RGB/RGBA or 2D grayscale, got {tuple(t.shape)}")
    # drop alpha if present
    if t.dim() == 3 and t.shape[2] == 4:
        t = t[..., :3]
    return t


# -------------- Public API --------------

def seam_carving(
    src: TensorOrPIL,
    size: Optional[Tuple[int, int]] = None,
    energy_mode: str = "backward",
    order: str = "width-first",
    keep_mask: Optional[TensorOrPIL] = None,
    drop_mask: Optional[TensorOrPIL] = None,
    step_ratio: float = 0.5,
) -> TensorOrPIL:
    """
    Content-aware resize via seam carving (torch-only).
    Inputs can be PIL or torch tensor (H,W[,C]) in [0,1]; returns same type as input.
    """
    src_is_pil = isinstance(src, Image.Image)
    src = _check_src(src)
    device = src.device if src.is_cuda else torch.device("cpu")
    src = src.to(device)

    if order not in _list_enum(OrderMode):
        raise ValueError(f"expect order in {_list_enum(OrderMode)}, got {order}")

    aux_energy: Optional[torch.Tensor] = None

    if keep_mask is not None:
        keep_mask_t = _check_mask(keep_mask, src.shape[:2]).to(device)
        aux_energy = torch.zeros(src.shape[:2], dtype=torch.float32, device=device)
        aux_energy[keep_mask_t] += KEEP_MASK_ENERGY

    # drop object first if given
    if drop_mask is not None:
        drop_mask_t = _check_mask(drop_mask, src.shape[:2]).to(device)
        if aux_energy is None:
            aux_energy = torch.zeros(src.shape[:2], dtype=torch.float32, device=device)
        aux_energy[drop_mask_t] -= DROP_MASK_ENERGY

        if order == OrderMode.HEIGHT_FIRST:
            src = _transpose_image(src)
            aux_energy = aux_energy.t().contiguous()

        num_seams = (aux_energy < 0).sum(dim=1).max().item()
        while num_seams > 0:
            src, aux_energy = _reduce_width(src, int(num_seams), energy_mode, aux_energy)
            num_seams = (aux_energy < 0).sum(dim=1).max().item()

        if order == OrderMode.HEIGHT_FIRST:
            src = _transpose_image(src)
            aux_energy = aux_energy.t().contiguous()

    if size is not None:
        width, height = size
        width = round(width)
        height = round(height)
        if width <= 0 or height <= 0:
            raise ValueError(f"expect positive target size, got {size}")

        if order == OrderMode.WIDTH_FIRST:
            src, aux_energy = _resize_width(src, width, energy_mode, aux_energy, step_ratio)
            src, aux_energy = _resize_height(src, height, energy_mode, aux_energy, step_ratio)
        else:
            src, aux_energy = _resize_height(src, height, energy_mode, aux_energy, step_ratio)
            src, aux_energy = _resize_width(src, width, energy_mode, aux_energy, step_ratio)

    # return same type as input
    return _to_pil(src) if src_is_pil else src
