# utils.py — numpy/scipy-free version compatible with comfyai.run
# - Replaces NumPy/SciPy morphology with pure-PyTorch ops.
# - Keeps same public functions/signatures.
# - Adds robust, device-safe mask expansion (square or cross/“tapered corners”).
#
# Drop-in replacement.

import os
from pathlib import Path

import torch
import torch.nn.functional as F

import folder_paths

FONTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")

SCRIPT_DIR = Path(__file__).parent
folder_paths.add_model_folder_path("luts", (SCRIPT_DIR / "luts").as_posix())
folder_paths.add_model_folder_path(
    "luts", (Path(folder_paths.models_dir) / "luts").as_posix()
)

# from https://github.com/pythongosssss/ComfyUI-Custom-Scripts
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


def min_(tensor_list):
    """Element-wise min of a list/tuple of tensors, clamped to [0, +inf)."""
    x = torch.stack(tensor_list)
    mn = x.min(dim=0)[0]
    return torch.clamp(mn, min=0)


def max_(tensor_list):
    """Element-wise max of a list/tuple of tensors, clamped to (-inf, 1]."""
    x = torch.stack(tensor_list)
    mx = x.max(dim=0)[0]
    return torch.clamp(mx, max=1)


# ---------------------------
# Torch-only morphology utils
# ---------------------------

def _ensure_bhw(mask: torch.Tensor) -> torch.Tensor:
    """Accepts (H,W) or (B,H,W) or (B,1,H,W) and returns (B,H,W) float in [0,1]."""
    m = mask
    if m.dim() == 4:
        # (B,1,H,W) or (B,C,H,W) -> take 1st channel
        m = m[:, 0, ...]
    elif m.dim() == 2:
        m = m.unsqueeze(0)  # -> (1,H,W)
    # now (B,H,W)
    if m.dtype != torch.float32:
        m = m.float()
    return m.clamp(0.0, 1.0)


def _odd(k: int) -> int:
    return 1 if k <= 0 else (k if k % 2 == 1 else k + 1)


def _dilate_square(mask_bhw: torch.Tensor, iterations: int) -> torch.Tensor:
    """Dilation with 3x3 square kernel via max_pool2d; repeated 'iterations' times."""
    if iterations <= 0:
        return mask_bhw
    k = 3
    pad = 1
    m = mask_bhw.unsqueeze(1)  # (B,1,H,W)
    for _ in range(iterations):
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=pad)
    return m.squeeze(1)


def _erode_square(mask_bhw: torch.Tensor, iterations: int) -> torch.Tensor:
    """Erosion with 3x3 square kernel via -max_pool2d(-x)."""
    if iterations <= 0:
        return mask_bhw
    k = 3
    pad = 1
    m = mask_bhw.unsqueeze(1)
    for _ in range(iterations):
        m = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=pad)
    return m.squeeze(1)


def _shift_like(m: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Shift (B,H,W) by (dx,dy) with zero fill."""
    B, H, W = m.shape
    out = torch.zeros_like(m)
    x_src0 = max(0, -dx)
    y_src0 = max(0, -dy)
    x_dst0 = max(0, dx)
    y_dst0 = max(0, dy)
    w = W - abs(dx)
    h = H - abs(dy)
    if w > 0 and h > 0:
        out[:, y_dst0:y_dst0 + h, x_dst0:x_dst0 + w] = m[:, y_src0:y_src0 + h, x_src0:x_src0 + w]
    return out


def _dilate_cross(mask_bhw: torch.Tensor, iterations: int) -> torch.Tensor:
    """Dilation with 3x3 cross (no corners). Do 4-neighbour max repeatedly."""
    if iterations <= 0:
        return mask_bhw
    m = mask_bhw
    for _ in range(iterations):
        candidates = [
            m,
            _shift_like(m, 1, 0),
            _shift_like(m, -1, 0),
            _shift_like(m, 0, 1),
            _shift_like(m, 0, -1),
        ]
        m = torch.maximum(torch.maximum(candidates[0], candidates[1]),
                          torch.maximum(candidates[2], torch.maximum(candidates[3], candidates[4])))
    return m


def _erode_cross(mask_bhw: torch.Tensor, iterations: int) -> torch.Tensor:
    """Erosion with 3x3 cross (no corners). Do 4-neighbour min repeatedly."""
    if iterations <= 0:
        return mask_bhw
    m = mask_bhw
    for _ in range(iterations):
        candidates = [
            m,
            _shift_like(m, 1, 0),
            _shift_like(m, -1, 0),
            _shift_like(m, 0, 1),
            _shift_like(m, 0, -1),
        ]
        # start from +inf-like tensor for min chain
        out = candidates[0]
        out = torch.minimum(out, candidates[1])
        out = torch.minimum(out, candidates[2])
        out = torch.minimum(out, candidates[3])
        out = torch.minimum(out, candidates[4])
        m = out
    return m


def expand_mask(mask, expand: int, tapered_corners: bool):
    """
    Torch-only mask grow/shrink.
    - mask: (H,W) or (B,H,W) or (B,1,H,W); float/bool/byte -> returns (B,H,W) float.
    - expand > 0 => dilate |expand| steps
    - expand < 0 => erode |expand| steps
    - tapered_corners=True uses cross-shaped structuring element (no corners),
      matching the old footprint=[[0,1,0],[1,1,1],[0,1,0]] behaviour.
    - tapered_corners=False uses 3x3 square.
    """
    m = _ensure_bhw(mask)
    iters = abs(int(expand))

    if iters == 0:
        return m

    if expand > 0:
        if tapered_corners:
            m = _dilate_cross(m, iters)
        else:
            m = _dilate_square(m, iters)
    else:
        if tapered_corners:
            m = _erode_cross(m, iters)
        else:
            m = _erode_square(m, iters)

    return m.clamp(0.0, 1.0)


def parse_string_to_list(s: str):
    """
    Parses strings like:
      "1, 2.5, 3...5+0.5" -> [1, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    Keeps decimal precision from 'step'.
    """
    elements = s.split(',')
    result = []

    def parse_number(v):
        v = v.strip()
        try:
            return float(v) if '.' in v else int(v)
        except ValueError:
            return 0

    def decimal_places(v):
        v = v.strip()
        return len(v.split('.')[1]) if '.' in v else 0

    for element in elements:
        element = element.strip()
        if '...' in element:
            start, rest = element.split('...')
            end, step = rest.split('+')
            decimals = decimal_places(step)
            start_v = parse_number(start)
            end_v = parse_number(end)
            step_v = parse_number(step)
            # normalize step direction
            if (start_v > end_v and step_v > 0) or (start_v < end_v and step_v < 0):
                step_v = -step_v
            current = start_v
            # inclusive range
            if step_v == 0:
                result.append(round(start_v, decimals))
            else:
                if start_v <= end_v:
                    while current <= end_v + 1e-12:
                        result.append(round(current, decimals))
                        current += step_v
                else:
                    while current >= end_v - 1e-12:
                        result.append(round(current, decimals))
                        current += step_v
        else:
            result.append(round(parse_number(element), decimal_places(element)))

    return result
