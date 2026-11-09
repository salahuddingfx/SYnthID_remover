"""
Spatial-Domain and Chrominance Purifiers.
Applies edge-preserving filtering, sub-pixel grid desynchronization,
and chrominance decorrelation to disrupt convolutional watermark detector alignment.
"""

import numpy as np
from PIL import Image, ImageFilter


def desynchronize_grid(img: Image.Image, scale_factor: float = 0.996) -> Image.Image:
    """
    Sub-pixel Spatial Grid Desynchronization.
    Downscales by a fraction of a percent with Lanczos resampling and restores dimensions.
    This shifts the spatial phase alignment of the convolutional kernels used in SynthID detectors.
    """
    w, h = img.size
    new_w = max(1, int(round(w * scale_factor)))
    new_h = max(1, int(round(h * scale_factor)))

    # Resample using high-order Lanczos filter to disrupt steganographic grid alignment
    resampled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    restored = resampled.resize((w, h), Image.Resampling.LANCZOS)
    return restored


def edge_preserving_denoise(img_arr: np.ndarray, diameter: int = 5, sigma_color: float = 12.0, sigma_space: float = 3.0) -> np.ndarray:
    """
    Fast Vectorized Edge-Preserving Bilateral Filtering.
    Smooths flat and fine-noise regions where steganographic perturbations reside,
    while locking onto and preserving high-contrast semantic edges.
    """
    orig_dtype = img_arr.dtype
    arr = img_arr.astype(np.float32)
    h, w, c = arr.shape
    radius = diameter // 2

    pad_arr = np.pad(arr, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    out = np.zeros_like(arr)

    # Spatial Gaussian weights
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    spatial_weights = np.exp(-(x**2 + y**2) / (2 * (sigma_space**2))).astype(np.float32)

    # Apply windowed bilateral estimation
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial_w = spatial_weights[dy + radius, dx + radius]
            neighbor = pad_arr[radius + dy : radius + dy + h, radius + dx : radius + dx + w, :]
            
            # Color intensity difference
            diff = neighbor - arr
            intensity_w = np.exp(-np.sum(diff**2, axis=-1, keepdims=True) / (2 * (sigma_color**2)))
            
            weight = spatial_w * intensity_w
            out += neighbor * weight
            pad_arr_w = weight if "norm" not in locals() else norm + weight
            norm = pad_arr_w

    out = out / (norm + 1e-7)
    out = np.clip(out, 0.0, 255.0)
    if orig_dtype == np.uint8:
        return np.round(out).astype(np.uint8)
    return out


def chroma_decorrelation(img_arr: np.ndarray, jitter_strength: float = 1.2) -> np.ndarray:
    """
    Perturbs chrominance (Cb, Cr) channels in YCbCr color space.
    SynthID leverages human visual insensitivity in color difference channels;
    subtle micro-dithering here severely degrades detector confidence.
    """
    orig_dtype = img_arr.dtype
    rgb = img_arr.astype(np.float32)

    # RGB to YCbCr conversion matrix
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    # Apply subtle zero-mean Gaussian jitter to chrominance
    cb_noise = np.random.normal(0, jitter_strength, size=cb.shape)
    cr_noise = np.random.normal(0, jitter_strength, size=cr.shape)

    cb_mod = np.clip(cb + cb_noise, 0.0, 255.0)
    cr_mod = np.clip(cr + cr_noise, 0.0, 255.0)

    # YCbCr back to RGB
    r_new = y + 1.402 * (cr_mod - 128.0)
    g_new = y - 0.344136 * (cb_mod - 128.0) - 0.714136 * (cr_mod - 128.0)
    b_new = y + 1.772 * (cb_mod - 128.0)

    out = np.stack([r_new, g_new, b_new], axis=-1)
    out = np.clip(out, 0.0, 255.0)

    if orig_dtype == np.uint8:
        return np.round(out).astype(np.uint8)
    return out
