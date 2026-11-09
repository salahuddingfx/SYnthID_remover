"""
Image Fidelity and Quality Metrics.
Calculates PSNR, SSIM, and MSE to mathematically verify that the output remains
visually indistinguishable from the input.
"""

import math
import numpy as np


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Mean Squared Error between two images."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Values > 40 dB indicate negligible to invisible perceptual differences.
    """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(max_val / math.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, k1: float = 0.01, k2: float = 0.03, l: float = 255.0) -> float:
    """
    Computes mean Structural Similarity Index (SSIM).
    Values > 0.98 indicate near-perfect perceptual similarity.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    c1 = (k1 * l) ** 2
    c2 = (k2 * l) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

    return float(numerator / denominator)


def evaluate_fidelity(orig: np.ndarray, cleaned: np.ndarray) -> dict:
    """Evaluates comprehensive fidelity metrics."""
    mse = calculate_mse(orig, cleaned)
    psnr = calculate_psnr(orig, cleaned)
    ssim = calculate_ssim(orig, cleaned)

    return {
        "mse": round(mse, 4),
        "psnr_db": round(psnr, 2) if psnr != float("inf") else "inf",
        "ssim": round(ssim, 4),
        "quality_preservation": "Excellent (Imperceptible)" if psnr > 40 else "Good" if psnr > 34 else "Moderate",
    }

