import numpy as np
import torch
from PIL import Image, ImageDraw
import random


def _ensure_numpy_mask(mask):
    # Convert torch tensors to numpy and squeeze batch dims if present
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    # If mask has extra dims like (1, H, W) or (N,1,H,W) squeeze appropriately
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    # If mask has shape (H, W, 1) squeeze the last dim
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.reshape(mask.shape[0], mask.shape[1])
    return mask


def show_pipe_masks_on_image(raw_image, outputs):
    """Return a PIL image with mask overlays applied.

    raw_image: PIL.Image
    outputs: dict with key 'masks' containing iterable of mask arrays/tensors
    """
    base = raw_image.convert("RGBA")
    composite = base.copy()

    for mask in outputs.get("masks", []):
        mask = _ensure_numpy_mask(mask)

        # If mask is probabilistic, threshold it
        if not np.issubdtype(mask.dtype, np.bool_):
            try:
                mask = mask > 0.5
            except Exception:
                mask = mask.astype(bool)

        h, w = mask.shape[-2:]

        # random color for each mask
        r, g, b = (random.randint(0, 255) for _ in range(3))
        alpha = int(0.6 * 255)

        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[mask] = (r, g, b, alpha)

        overlay_img = Image.fromarray(overlay, mode="RGBA")

        # If mask resolution differs from image, resize using nearest neighbour
        if overlay_img.size != base.size:
            overlay_img = overlay_img.resize(base.size, resample=Image.NEAREST)

        composite = Image.alpha_composite(composite, overlay_img)

    return composite.convert("RGB")


# Provide a simple box drawing utility if needed by other code
def show_boxes_on_image(raw_image, boxes):
    img = raw_image.convert("RGBA").copy()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
    return img.convert("RGB")

def show_mask_on_image(raw_image, mask, return_image=False):
    """Overlay a single mask on the raw_image and either display or return a PIL image.

    raw_image: PIL.Image
    mask: tensor/ndarray of shape (H, W) or (1, H, W) etc.
    return_image: if True, return the PIL.Image, otherwise call Image.show()
    """
    mask = _ensure_numpy_mask(mask)

    # If mask is probabilistic, threshold it
    if not np.issubdtype(mask.dtype, np.bool_):
        try:
            mask = mask > 0.5
        except Exception:
            mask = mask.astype(bool)

    h, w = mask.shape[-2:]

    # Choose a fixed color for single-mask display
    r, g, b = (30, 144, 255)
    alpha = int(0.6 * 255)

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask] = (r, g, b, alpha)

    overlay_img = Image.fromarray(overlay, mode="RGBA")

    base = raw_image.convert("RGBA")
    if overlay_img.size != base.size:
        overlay_img = overlay_img.resize(base.size, resample=Image.NEAREST)

    composite = Image.alpha_composite(base, overlay_img).convert("RGB")

    if return_image:
        return composite
    else:
        composite.show()
        return None