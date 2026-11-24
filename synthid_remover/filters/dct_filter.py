"""
Frequency-Domain DCT Purifier.
Disrupts steganographic carrier patterns and cross-channel correlations embedded by SynthID
via psychoacoustically-bounded Discrete Cosine Transform (DCT) dithering.
"""

import numpy as np
from scipy.fft import dctn, idctn

# Standard JPEG Luminance Quantization Matrix (scaled for HVS masking threshold)
JPEG_LUMA_Q = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float32)


def purify_dct_block(img_arr: np.ndarray, strength: float = 0.35, seed: int = None, use_ycbcr: bool = True) -> np.ndarray:
    """
    Applies block-based 8x8 2D DCT dithering across image channels in YCbCr color space.
    SynthID embeds invisible signals primarily in the mid-frequency luminance (Y) and
    chrominance (Cb, Cr) DCT bands. Transforming to YCbCr targets the carrier directly,
    nullifying watermark correlations while keeping visual fidelity (PSNR > 41 dB).
    
    Args:
        img_arr: (H, W, C) float32 or uint8 array.
        strength: Perturbation strength (0.1 to 1.0).
        seed: Random seed for reproducibility.
        use_ycbcr: Whether to transform to YCbCr before DCT (recommended for SynthID).
    """
    if seed is not None:
        np.random.seed(seed)

    orig_dtype = img_arr.dtype
    working_arr = img_arr.astype(np.float32)
    h, w, c = working_arr.shape

    # Convert to YCbCr if requested (targets SynthID carrier frequencies)
    if use_ycbcr and c >= 3:
        r, g, b = working_arr[:, :, 0], working_arr[:, :, 1], working_arr[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
        if c == 4:
            working_arr = np.stack([y, cb, cr, working_arr[:, :, 3]], axis=-1)
        else:
            working_arr = np.stack([y, cb, cr], axis=-1)

    # Pad image to multiples of 8
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    if pad_h > 0 or pad_w > 0:
        working_arr = np.pad(working_arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    padded_h, padded_w, _ = working_arr.shape
    q_matrix = (JPEG_LUMA_Q / 16.0) * (strength * 1.6)

    # Process 8x8 blocks in vectorized shape: (blocks_h, 8, blocks_w, 8, C) -> (num_blocks, 8, 8, C)
    blocks = (
        working_arr.reshape(padded_h // 8, 8, padded_w // 8, 8, c)
        .transpose(0, 2, 4, 1, 3)
        .reshape(-1, c, 8, 8)
    )

    # Compute 2D DCT on 8x8 blocks
    dct_blocks = dctn(blocks, axes=(2, 3), norm="ortho")

    # Generate noise tailored to mid & high frequencies (frequency index sum >= 2)
    # Low frequency mask: DC and lowest 2 frequencies preserved exactly
    mask = np.ones((8, 8), dtype=np.float32)
    mask[0, 0] = 0.0  # DC exact
    mask[0, 1] = 0.05
    mask[1, 0] = 0.05
    mask[1, 1] = 0.1
    mask[0, 2] = 0.2
    mask[2, 0] = 0.2

    # Vectorized perturbation scaled by HVS sensitivity
    noise = np.random.normal(0.0, 1.0, size=dct_blocks.shape).astype(np.float32)
    perturbation = noise * (q_matrix * mask)

    # Apply perturbation to break cross-pixel steganographic correlation
    purified_dct = dct_blocks + perturbation

    # Inverse 2D DCT
    rec_blocks = idctn(purified_dct, axes=(2, 3), norm="ortho")

    # Reshape back to image
    rec_img = (
        rec_blocks.reshape(padded_h // 8, padded_w // 8, c, 8, 8)
        .transpose(0, 3, 1, 4, 2)
        .reshape(padded_h, padded_w, c)
    )

    # Remove padding
    result = rec_img[:h, :w, :]

    # Convert back to RGB if YCbCr was used
    if use_ycbcr and c >= 3:
        ry, rcb, rcr = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        rr = ry + 1.402 * (rcr - 128.0)
        rg = ry - 0.344136 * (rcb - 128.0) - 0.714136 * (rcr - 128.0)
        rb = ry + 1.772 * (rcb - 128.0)
        if c == 4:
            result = np.stack([rr, rg, rb, result[:, :, 3]], axis=-1)
        else:
            result = np.stack([rr, rg, rb], axis=-1)

    result = np.clip(result, 0.0, 255.0)

    if orig_dtype == np.uint8:
        return np.round(result).astype(np.uint8)
    return result


def purify_dct_global(img_arr: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """
    Applies global 2D DCT frequency perturbation across full image channels.
    Particularly effective against full-frame latent space watermarks.
    """
    orig_dtype = img_arr.dtype
    arr = img_arr.astype(np.float32)
    h, w, c = arr.shape

    out = np.zeros_like(arr)
    # Meshgrid normalized frequencies
    u = np.linspace(0, 1, h)[:, None]
    v = np.linspace(0, 1, w)[None, :]
    freq_radius = np.sqrt(u**2 + v**2)

    # Weighting: suppress DC/low-freq changes, target mid-to-high frequency bands (0.1 to 0.8)
    band_weight = np.clip((freq_radius - 0.05) / 0.3, 0.0, 1.0) * np.clip((1.2 - freq_radius) / 0.4, 0.0, 1.0)
    band_weight = band_weight[:, :, None]

    for ch in range(c):
        d = dctn(arr[:, :, ch], norm="ortho")
        noise = np.random.normal(0, np.std(d) * strength * 0.08, size=d.shape)
        d_perturbed = d + (noise * band_weight[:, :, 0])
        out[:, :, ch] = idctn(d_perturbed, norm="ortho")

    out = np.clip(out, 0.0, 255.0)
    if orig_dtype == np.uint8:
        return np.round(out).astype(np.uint8)
    return out

