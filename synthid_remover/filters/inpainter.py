"""
Visible Watermark Detector and Inpainter.
Detects and cleanly inpaints visible corner badges (such as Google Gemini/Veo sparkle stamps)
that often accompany invisible SynthID watermarks.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageFilter


def detect_corner_watermark_region(img_arr: np.ndarray, corner: str = "bottom_right", patch_ratio: float = 0.12) -> Tuple[int, int, int, int]:
    """
    Returns (ymin, ymax, xmin, xmax) for the suspected corner watermark region.
    """
    h, w = img_arr.shape[:2]
    patch_h = int(h * patch_ratio)
    patch_w = int(w * patch_ratio)

    if corner == "bottom_right":
        return h - patch_h, h, w - patch_w, w
    elif corner == "bottom_left":
        return h - patch_h, h, 0, patch_w
    elif corner == "top_right":
        return 0, patch_h, w - patch_w, w
    elif corner == "top_left":
        return 0, patch_h, 0, patch_w
    else:
        raise ValueError(f"Unknown corner: {corner}")


def generate_badge_mask(patch: np.ndarray, brightness_threshold: float = 230.0) -> np.ndarray:
    """
    Creates a binary inpainting mask for high-intensity translucent badges
    (like the Gemini white/semi-transparent sparkle overlay).
    """
    if patch.ndim == 3:
        # Luminance
        luma = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
    else:
        luma = patch.astype(np.float32)

    # Local contrast deviation
    local_mean = np.mean(luma)
    local_std = np.std(luma)

    # Pixels significantly brighter than local surroundings or above threshold
    mask = (luma > brightness_threshold) | (luma > local_mean + 1.8 * local_std)
    return mask.astype(np.uint8)


def inpaint_patch_telea_approx(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Pure-numpy fast edge-guided inpainting (approximate Telea/biharmonic method).
    Replaces masked watermark pixels using weighted boundary neighbor interpolation.
    """
    inpainted = patch.copy().astype(np.float32)
    h, w, c = patch.shape

    # Dilate mask slightly to cover watermark anti-aliasing halo
    dilated_mask = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            dilated_mask |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)

    unmasked_indices = np.where(dilated_mask == 0)
    masked_indices = np.where(dilated_mask == 1)

    if len(unmasked_indices[0]) == 0 or len(masked_indices[0]) == 0:
        return patch

    # Fast multi-pass diffusion fill across mask
    kernel = np.array([[0.05, 0.2, 0.05],
                       [0.2,  0.0, 0.2],
                       [0.05, 0.2, 0.05]], dtype=np.float32)

    pad_patch = np.pad(inpainted, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    pad_mask = np.pad(dilated_mask, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    # 15 iterations of smooth laplacian boundary propagation
    for _ in range(15):
        for ch in range(c):
            conv = (
                pad_patch[:-2, 1:-1, ch] * kernel[0, 1] +
                pad_patch[2:, 1:-1, ch] * kernel[2, 1] +
                pad_patch[1:-1, :-2, ch] * kernel[1, 0] +
                pad_patch[1:-1, 2:, ch] * kernel[1, 2] +
                pad_patch[:-2, :-2, ch] * kernel[0, 0] +
                pad_patch[:-2, 2:, ch] * kernel[0, 2] +
                pad_patch[2:, :-2, ch] * kernel[2, 0] +
                pad_patch[2:, 2:, ch] * kernel[2, 2]
            )
            conv_norm = conv / (kernel.sum() + 1e-6)
            inpainted[masked_indices[0], masked_indices[1], ch] = conv_norm[masked_indices[0], masked_indices[1]]
            pad_patch[1:-1, 1:-1, ch] = inpainted[:, :, ch]

    return np.clip(inpainted, 0, 255).astype(np.uint8)


def remove_visible_badge(img: Image.Image, corner: str = "bottom_right") -> Image.Image:
    """
    Detects and inpaints visible watermark badge in the specified corner.
    """
    arr = np.array(img.convert("RGB"))
    ymin, ymax, xmin, xmax = detect_corner_watermark_region(arr, corner=corner)
    patch = arr[ymin:ymax, xmin:xmax, :]

    mask = generate_badge_mask(patch)
    # Only inpaint if suspect badge pixels are detected (between 0.5% and 35% of the patch)
    mask_ratio = np.mean(mask)
    if 0.005 < mask_ratio < 0.35:
        clean_patch = inpaint_patch_telea_approx(patch, mask)
        arr[ymin:ymax, xmin:xmax, :] = clean_patch
        return Image.fromarray(arr)

    return img

