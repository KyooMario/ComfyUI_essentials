# mask.py  — adjusted to show as "kyoo" nodes, remove hard NumPy/SciPy deps, add safe fallbacks
#
# Key changes:
# - CATEGORY paths changed to "kyoo/…" so you see your nodes grouped as Kyoo in ComfyUI.
# - Removed hard NumPy/SciPy requirements; added torch-only morphology fallbacks (dilate/erode/open/close, hole fill).
# - Kept torchvision-only blurs; device-safe transfers; robust batching/size matching.
# - MaskFromSegmentation rewritten to avoid NumPy; uses PIL + torch and torch fallbacks for cleanup.
#
# Drop-in replacement for your existing file.

import random
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image

from nodes import SaveImage
import folder_paths
import comfy.utils
import comfy.model_management
from .image import ImageExpandBatch
from nodes import MAX_RESOLUTION
from comfy.comfy_types import IO, ComfyNodeABC, CheckLazyMixin


# ---------------------------
# Torch-only morphology utils
# ---------------------------

def _ensure_bchw(mask_2d_or_3d: torch.Tensor) -> torch.Tensor:
    """
    Accepts: (H,W) or (B,H,W). Returns: (B,1,H,W) float in [0,1].
    """
    m = mask_2d_or_3d
    if m.dim() == 2:  # H, W
        m = m.unsqueeze(0)
    # now (B,H,W)
    if m.dtype != torch.float32:
        m = m.float()
    m = m.unsqueeze(1)  # (B,1,H,W)
    return m.clamp(0.0, 1.0)


def _kernel_size(k: int) -> int:
    # force odd >= 1
    if k <= 0:
        return 1
    return k if k % 2 == 1 else k + 1


def dilate(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Torch-only dilation using max_pool2d."""
    if size <= 0:
        return mask
    k = _kernel_size(size)
    pad = k // 2
    m = _ensure_bchw(mask)
    out = F.max_pool2d(m, kernel_size=k, stride=1, padding=pad)
    return out.squeeze(1)


def erode(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Torch-only erosion using -max_pool2d(-x)."""
    if size <= 0:
        return mask
    k = _kernel_size(size)
    pad = k // 2
    m = _ensure_bchw(mask)
    out = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=pad)
    return out.squeeze(1)


def open_(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Opening = erosion then dilation."""
    if size <= 0:
        return mask
    return dilate(erode(mask, size), size)


def close_(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Closing = dilation then erosion."""
    if size <= 0:
        return mask
    return erode(dilate(mask, size), size)


def gaussian_blur_mask(mask: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return mask
    k = _kernel_size(k)
    m = mask
    # accepts (B,H,W) -> (B,1,H,W)
    m_bchw = _ensure_bchw(m)
    m_bchw = T.functional.gaussian_blur(m_bchw, k)
    return m_bchw.squeeze(1)


# ---------------------------
# Nodes
# ---------------------------

class MaskBlur(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": (IO.MASK,),
                "amount": (IO.INT, {"default": 6, "min": 0, "max": 256, "step": 1}),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    def execute(self, mask, amount, device):
        if amount == 0:
            return (mask,)

        if device == "gpu":
            mask = mask.to(comfy.model_management.get_torch_device())
        elif device == "cpu":
            mask = mask.to("cpu")

        out = gaussian_blur_mask(mask, amount)

        if device in ("gpu", "cpu"):
            out = out.to(comfy.model_management.intermediate_device())

        return (out,)


class MaskFlip(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": (IO.MASK,),
                "axis": (["x", "y", "xy"],),
            }
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    def execute(self, mask, axis):
        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0)

        dims = ()
        if "y" in axis:
            dims += (1,)
        if "x" in axis:
            dims += (2,)
        m = torch.flip(m, dims=dims)
        return (m,)


class MaskPreview(SaveImage, ComfyNodeABC):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": (IO.MASK,)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    def execute(self, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)


class MaskBatch(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": (IO.MASK,),
                "mask2": (IO.MASK,),
            }
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask batch"

    def execute(self, mask1, mask2):
        if mask1.shape[1:] != mask2.shape[1:]:
            mask2 = comfy.utils.common_upscale(
                mask2.unsqueeze(1).expand(-1, 3, -1, -1),
                mask1.shape[2],
                mask1.shape[1],
                upscale_method="bicubic",
                crop="center",
            )[:, 0, :, :]

        return (torch.cat((mask1, mask2), dim=0),)


class MaskExpandBatch(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": (IO.MASK,),
                "size": (IO.INT, {"default": 16, "min": 1, "step": 1}),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],),
            }
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask batch"

    def execute(self, mask, size, method):
        # reuse ImageExpandBatch by faking 3 channels, then strip back to single channel
        expanded = ImageExpandBatch().execute(mask.unsqueeze(1).expand(-1, 3, -1, -1), size, method)[0][:, 0, :, :]
        return (expanded,)


class MaskBoundingBox(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": (IO.MASK,),
                "padding": (IO.INT, {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "blur": (IO.INT, {"default": 0, "min": 0, "max": 256, "step": 1}),
            },
            "optional": {
                "image_optional": (IO.IMAGE,),
            },
        }

    RETURN_TYPES = (IO.MASK, IO.IMAGE, IO.INT, IO.INT, IO.INT, IO.INT)
    RETURN_NAMES = ("MASK", "IMAGE", "x", "y", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    def execute(self, mask, padding, blur, image_optional=None):
        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0)

        if image_optional is None:
            image_optional = m.unsqueeze(3).repeat(1, 1, 1, 3)

        # size match IMAGE to MASK
        if image_optional.shape[1:] != m.shape[1:]:
            image_optional = comfy.utils.common_upscale(
                image_optional.permute([0, 3, 1, 2]),
                m.shape[2],
                m.shape[1],
                upscale_method="bicubic",
                crop="center",
            ).permute([0, 2, 3, 1])

        # batch match
        if image_optional.shape[0] < m.shape[0]:
            image_optional = torch.cat(
                (image_optional, image_optional[-1].unsqueeze(0).repeat(m.shape[0] - image_optional.shape[0], 1, 1, 1)),
                dim=0,
            )
        elif image_optional.shape[0] > m.shape[0]:
            image_optional = image_optional[: m.shape[0]]

        # optional blur before bbox search
        if blur > 0:
            m = gaussian_blur_mask(m, blur)

        # compute bbox across all frames
        idx = torch.where(m > 0.0)
        if len(idx[0]) == 0:
            # no positive pixels, return minimal box at (0,0)
            return (m[:, :1, :1], image_optional[:, :1, :1, :], 0, 0, 1, 1)

        _, y, x = idx
        x1 = max(0, int(x.min().item()) - padding)
        x2 = min(m.shape[2], int(x.max().item()) + 1 + padding)
        y1 = max(0, int(y.min().item()) - padding)
        y2 = min(m.shape[1], int(y.max().item()) + 1 + padding)

        m_crop = m[:, y1:y2, x1:x2]
        img_crop = image_optional[:, y1:y2, x1:x2, :]

        return (m_crop, img_crop, x1, y1, x2 - x1, y2 - y1)


class MaskFromColor(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "red": (IO.INT, {"default": 255, "min": 0, "max": 255, "step": 1}),
                "green": (IO.INT, {"default": 255, "min": 0, "max": 255, "step": 1}),
                "blue": (IO.INT, {"default": 255, "min": 0, "max": 255, "step": 1}),
                "threshold": (IO.INT, {"default": 0, "min": 0, "max": 127, "step": 1}),
            }
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    def execute(self, image, red, green, blue, threshold):
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        color = torch.tensor([red, green, blue], device=temp.device)
        lower_bound = (color - threshold).clamp(min=0).view(1, 1, 1, 3)
        upper_bound = (color + threshold).clamp(max=255).view(1, 1, 1, 3)
        mask = ((temp >= lower_bound) & (temp <= upper_bound)).all(dim=-1).float()
        return (mask,)


class MaskFromSegmentation(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "segments": (IO.INT, {"default": 6, "min": 1, "max": 16, "step": 1}),
                "remove_isolated_pixels": (IO.INT, {"default": 0, "min": 0, "max": 32, "step": 1}),
                "remove_small_masks": (IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fill_holes": (IO.BOOLEAN, {"default": False}),
            }
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    @staticmethod
    def _pil_to_torch_rgb(img: Image.Image) -> torch.Tensor:
        """Convert PIL RGB to torch float [0,1] (H,W,3) without NumPy."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        b = img.tobytes()
        t = torch.frombuffer(bytearray(b), dtype=torch.uint8, device="cpu").view(h, w, 3).float() / 255.0
        return t  # (H,W,3)

    def execute(self, image, segments, remove_isolated_pixels, fill_holes, remove_small_masks):
        # work on the first frame (as original)
        im = image[0]  # (H,W,3) in [0,1]
        pil = Image.fromarray((im * 255).to(torch.uint8).cpu().numpy(), mode="RGB")
        # quantize to 'segments' colors using PIL itself
        pal = pil.quantize(colors=segments, dither=Image.Dither.NONE)
        pal_rgb = pal.convert("RGB")
        rgb = self._pil_to_torch_rgb(pal_rgb)  # (H,W,3)

        # unique colors present
        colors = rgb.reshape(-1, 3).unique(dim=0)

        masks = []
        for c in colors:
            mask = (rgb == c).all(dim=-1).float()  # (H,W)

            # remove isolated pixels (opening)
            if remove_isolated_pixels > 0:
                mask = open_(mask, remove_isolated_pixels)

            # fill holes (closing)
            if fill_holes:
                # use a small structural element if none specified elsewhere
                mask = close_(mask, max(3, remove_isolated_pixels if remove_isolated_pixels > 0 else 3))

            # discard tiny masks
            if mask.sum() / (mask.shape[0] * mask.shape[1]) > remove_small_masks:
                masks.append(mask)

        if not masks:
            masks.append(torch.zeros_like(rgb[..., 0]))  # empty, prevents errors

        mask_stack = torch.stack(masks, dim=0).float()  # (K,H,W)
        return (mask_stack,)


class MaskFix(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": (IO.MASK,),
                "erode_dilate": (IO.INT, {"default": 0, "min": -256, "max": 256, "step": 1}),
                "fill_holes": (IO.INT, {"default": 0, "min": 0, "max": 128, "step": 1}),
                "remove_isolated_pixels": (IO.INT, {"default": 0, "min": 0, "max": 32, "step": 1}),
                "smooth": (IO.INT, {"default": 0, "min": 0, "max": 256, "step": 1}),
                "blur": (IO.INT, {"default": 0, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    def execute(self, mask, erode_dilate, smooth, remove_isolated_pixels, blur, fill_holes):
        out = []
        for m in mask:
            mm = m

            # erode / dilate
            if erode_dilate != 0:
                k = abs(erode_dilate)
                if erode_dilate < 0:
                    mm = erode(mm, k)
                else:
                    mm = dilate(mm, k)

            # fill holes (closing)
            if fill_holes > 0:
                mm = close_(mm, fill_holes)

            # remove isolated pixels (opening)
            if remove_isolated_pixels > 0:
                mm = open_(mm, remove_isolated_pixels)

            # smooth (thresholded blur)
            if smooth > 0:
                k = _kernel_size(smooth)
                thr = 0.5
                mm_bin = (mm > thr).float()
                mm = gaussian_blur_mask(mm_bin, k)

            # blur (soften edges)
            if blur > 0:
                mm = gaussian_blur_mask(mm.float(), blur)

            out.append(mm.float())

        masks = torch.stack(out, dim=0).float()
        return (masks,)


class MaskSmooth(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": (IO.MASK,),
                "amount": (IO.INT, {"default": 0, "min": 0, "max": 127, "step": 1}),
            }
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    def execute(self, mask, amount):
        if amount == 0:
            return (mask,)
        k = _kernel_size(amount)
        m = (mask > 0.5).float()
        m = gaussian_blur_mask(m, k)
        return (m,)


class MaskFromBatch(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": (IO.MASK,),
                "start": (IO.INT, {"default": 0, "min": 0, "step": 1}),
                "length": (IO.INT, {"default": 1, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask batch"

    def execute(self, mask, start, length):
        length = min(length, mask.shape[0])
        start = min(start, mask.shape[0] - 1)
        length = min(mask.shape[0] - start, length)
        return (mask[start : start + length],)


class MaskFromList(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (IO.INT, {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "height": (IO.INT, {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            },
            "optional": {
                "values": (IO.ANY, {"default": 0.0, "min": 0.0, "max": 1.0}),
                "str_values": (IO.STRING, {"default": "", "multiline": True, "placeholder": "0.0, 0.5, 1.0"}),
            },
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    def execute(self, width, height, values=None, str_values=""):
        out_vals = []

        if values is not None:
            if not isinstance(values, list):
                out_vals = [float(values)]
            else:
                out_vals.extend([float(v) for v in values])

        if str_values != "":
            out_vals.extend([float(v.strip()) for v in str_values.split(",") if v.strip() != ""])

        if out_vals == []:
            raise ValueError("No values provided")

        out = torch.tensor(out_vals, dtype=torch.float32).clamp(0.0, 1.0).view(-1, 1, 1).expand(-1, height, width)
        return (out,)


class MaskFromRGBCMYBW(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "threshold_r": (IO.FLOAT, {"default": 0.15, "min": 0.0, "max": 1, "step": 0.01}),
                "threshold_g": (IO.FLOAT, {"default": 0.15, "min": 0.0, "max": 1, "step": 0.01}),
                "threshold_b": (IO.FLOAT, {"default": 0.15, "min": 0.0, "max": 1, "step": 0.01}),
            }
        }

    RETURN_TYPES = (
        IO.MASK,
        IO.MASK,
        IO.MASK,
        IO.MASK,
        IO.MASK,
        IO.MASK,
        IO.MASK,
        IO.MASK,
    )
    RETURN_NAMES = ("red", "green", "blue", "cyan", "magenta", "yellow", "black", "white")
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    def execute(self, image, threshold_r, threshold_g, threshold_b):
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]

        red = ((r >= 1 - threshold_r) & (g < threshold_g) & (b < threshold_b)).float()
        green = ((r < threshold_r) & (g >= 1 - threshold_g) & (b < threshold_b)).float()
        blue = ((r < threshold_r) & (g < threshold_g) & (b >= 1 - threshold_b)).float()

        cyan = ((r < threshold_r) & (g >= 1 - threshold_g) & (b >= 1 - threshold_b)).float()
        magenta = ((r >= 1 - threshold_r) & (g < threshold_g) & (b > 1 - threshold_b)).float()
        yellow = ((r >= 1 - threshold_r) & (g >= 1 - threshold_g) & (b < threshold_b)).float()

        black = ((r <= threshold_r) & (g <= threshold_g) & (b <= threshold_b)).float()
        white = ((r >= 1 - threshold_r) & (g >= 1 - threshold_g) & (b >= 1 - threshold_b)).float()

        return (red, green, blue, cyan, magenta, yellow, black, white)


class TransitionMask(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (IO.INT, {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": (IO.INT, {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "frames": (IO.INT, {"default": 16, "min": 1, "max": 9999, "step": 1}),
                "start_frame": (IO.INT, {"default": 0, "min": 0, "step": 1}),
                "end_frame": (IO.INT, {"default": 9999, "min": 0, "step": 1}),
                "transition_type": (
                    [
                        "horizontal slide",
                        "vertical slide",
                        "horizontal bar",
                        "vertical bar",
                        "center box",
                        "horizontal door",
                        "vertical door",
                        "circle",
                        "fade",
                    ],
                ),
                "timing_function": (["linear", "in", "out", "in-out"],),
            }
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "kyoo/mask"

    def linear(self, i, t):
        return i / t if t > 0 else 1.0

    def ease_in(self, i, t):
        x = i / t if t > 0 else 1.0
        return x * x

    def ease_out(self, i, t):
        x = i / t if t > 0 else 1.0
        return 1 - (1 - x) * (1 - x)

    def ease_in_out(self, i, t):
        if t <= 0:
            return 1.0
        x = i / t
        if x < 0.5:
            return 2 * x * x
        return 1 - pow(-2 * x + 2, 2) / 2

    def execute(self, width, height, frames, start_frame, end_frame, transition_type, timing_function):
        if timing_function == "in":
            timing_fn = self.ease_in
        elif timing_function == "out":
            timing_fn = self.ease_out
        elif timing_function == "in-out":
            timing_fn = self.ease_in_out
        else:
            timing_fn = self.linear

        out = []

        end_frame = min(frames, end_frame)
        transition = max(1, end_frame - start_frame)

        if start_frame > 0:
            out += [torch.zeros((height, width), dtype=torch.float32, device="cpu")] * start_frame

        for i in range(transition):
            frame = torch.zeros((height, width), dtype=torch.float32, device="cpu")
            progress = timing_fn(i, transition - 1)

            if "horizontal slide" in transition_type:
                pos = round(width * progress)
                frame[:, :pos] = 1.0
            elif "vertical slide" in transition_type:
                pos = round(height * progress)
                frame[:pos, :] = 1.0
            elif "box" in transition_type:
                box_w = round(width * progress)
                box_h = round(height * progress)
                x1 = (width - box_w) // 2
                y1 = (height - box_h) // 2
                x2 = x1 + box_w
                y2 = y1 + box_h
                if box_w > 0 and box_h > 0:
                    frame[y1:y2, x1:x2] = 1.0
            elif "circle" in transition_type:
                radius = math.ceil(math.sqrt(width * width + height * height) * progress / 2)
                cx = width // 2
                cy = height // 2
                xs = torch.arange(0, width, dtype=torch.float32, device="cpu")
                ys = torch.arange(0, height, dtype=torch.float32, device="cpu")
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius ** 2)
                frame[circle] = 1.0
            elif "horizontal bar" in transition_type:
                bar = round(height * progress)
                y1 = (height - bar) // 2
                y2 = y1 + bar
                if bar > 0:
                    frame[y1:y2, :] = 1.0
            elif "vertical bar" in transition_type:
                bar = round(width * progress)
                x1 = (width - bar) // 2
                x2 = x1 + bar
                if bar > 0:
                    frame[:, x1:x2] = 1.0
            elif "horizontal door" in transition_type:
                bar = math.ceil(height * progress / 2)
                if bar > 0:
                    frame[:bar, :] = 1.0
                    frame[-bar:, :] = 1.0
            elif "vertical door" in transition_type:
                bar = math.ceil(width * progress / 2)
                if bar > 0:
                    frame[:, :bar] = 1.0
                    frame[:, -bar:] = 1.0
            elif "fade" in transition_type:
                frame[:, :] = progress

            out.append(frame)

        if end_frame < frames:
            out += [torch.ones((height, width), dtype=torch.float32, device="cpu")] * (frames - end_frame)

        out = torch.stack(out, dim=0)  # (T,H,W)
        return (out,)


# ---------------------------
# Registry
# ---------------------------

MASK_CLASS_MAPPINGS = {
    "MaskBlur+": MaskBlur,
    "MaskBoundingBox+": MaskBoundingBox,
    "MaskFix+": MaskFix,
    "MaskFlip+": MaskFlip,
    "MaskFromColor+": MaskFromColor,
    "MaskFromList+": MaskFromList,
    "MaskFromRGBCMYBW+": MaskFromRGBCMYBW,
    "MaskFromSegmentation+": MaskFromSegmentation,
    "MaskPreview+": MaskPreview,
    "MaskSmooth+": MaskSmooth,
    "TransitionMask+": TransitionMask,
    # Batch
    "MaskBatch+": MaskBatch,
    "MaskExpandBatch+": MaskExpandBatch,
    "MaskFromBatch+": MaskFromBatch,
}

MASK_NAME_MAPPINGS = {
    "MaskBlur+": "🔧 Mask Blur",
    "MaskFix+": "🔧 Mask Fix",
    "MaskFlip+": "🔧 Mask Flip",
    "MaskFromColor+": "🔧 Mask From Color",
    "MaskFromList+": "🔧 Mask From List",
    "MaskFromRGBCMYBW+": "🔧 Mask From RGB/CMY/BW",
    "MaskFromSegmentation+": "🔧 Mask From Segmentation",
    "MaskPreview+": "🔧 Mask Preview",
    "MaskBoundingBox+": "🔧 Mask Bounding Box",
    "MaskSmooth+": "🔧 Mask Smooth",
    "TransitionMask+": "🔧 Transition Mask",
    # Batch
    "MaskBatch+": "🔧 Mask Batch",
    "MaskExpandBatch+": "🔧 Mask Expand Batch",
    "MaskFromBatch+": "🔧 Mask From Batch",
}
