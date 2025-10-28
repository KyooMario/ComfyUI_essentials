# segmentation.py — numpy-free (torch/PIL only)

import torch
import torchvision.transforms.v2 as T
import torch.nn.functional as F

from comfy.comfy_types import IO, ComfyNodeABC, CheckLazyMixin

from .utils import expand_mask


class LoadCLIPSegModels(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("CLIP_SEG",)
    FUNCTION = "execute"
    CATEGORY = "essentials/segmentation"

    def execute(self):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        model.eval()  # inference mode

        return ((processor, model),)


class ApplyCLIPSeg(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_seg": ("CLIP_SEG",),
                "image": (IO.IMAGE,),
                "prompt": (IO.STRING, {"multiline": False, "default": ""}),
                "threshold": (IO.FLOAT, {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth": (IO.INT, {"default": 9, "min": 0, "max": 32, "step": 1}),
                "dilate": (IO.INT, {"default": 0, "min": -32, "max": 32, "step": 1}),
                "blur": (IO.INT, {"default": 0, "min": 0, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = (IO.MASK,)
    FUNCTION = "execute"
    CATEGORY = "essentials/segmentation"

    def execute(self, image, clip_seg, prompt, threshold, smooth, dilate, blur):
        """
        image: (B, H, W, C) float in [0,1]
        Returns mask: (B, H, W) float in [0,1]
        """
        processor, model = clip_seg

        # Work frame-by-frame to avoid NumPy dependency; use PIL via torchvision
        outputs = []
        to_pil = T.ToPILImage()  # expects CHW
        with torch.no_grad():
            for i in image:  # i: (H, W, C)
                pil_img = to_pil(i.permute(2, 0, 1).clamp(0, 1).cpu())
                inputs = processor(text=prompt, images=[pil_img], return_tensors="pt")
                out = model(**inputs)
                # out.logits: (batch=1, classes=1, H', W')
                logit = out.logits[0, 0]              # (H', W')
                prob = torch.sigmoid(logit)           # (H', W')
                mask = (prob > threshold).float()     # (H', W')
                outputs.append(mask)

        outputs = torch.stack(outputs, dim=0)  # (B, H', W')

        # Optional smoothing on mask
        if smooth > 0:
            if smooth % 2 == 0:
                smooth += 1
            outputs = T.functional.gaussian_blur(outputs.unsqueeze(1), smooth).squeeze(1)

        # Optional morphological expand/shrink (torch-only impl from utils.expand_mask)
        if dilate != 0:
            outputs = expand_mask(outputs, dilate, tapered_corners=True)

        # Optional blur
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            outputs = T.functional.gaussian_blur(outputs.unsqueeze(1), blur).squeeze(1)

        # Resize back to original spatial size
        outputs = F.interpolate(
            outputs.unsqueeze(1),
            size=(image.shape[1], image.shape[2]),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

        return (outputs.clamp(0.0, 1.0),)


SEG_CLASS_MAPPINGS = {
    "ApplyCLIPSeg+": ApplyCLIPSeg,
    "LoadCLIPSegModels+": LoadCLIPSegModels,
}

SEG_NAME_MAPPINGS = {
    "ApplyCLIPSeg+": "🔧 Apply CLIPSeg",
    "LoadCLIPSegModels+": "🔧 Load CLIPSeg Models",
}
