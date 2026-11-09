"""
Inspector and Spectral Analyzer for AI Images.
Analyzes metadata (EXIF, C2PA, XMP) and computes 2D FFT / DCT frequency spectra
along with residual difference heatmaps.
"""

import os
import struct
import numpy as np
from PIL import Image

from synthid_remover.metadata.cleaner import C2PA_BOX_UUID


def inspect_image_metadata(image_path: str) -> dict:
    """
    Scans image for embedded metadata, AI provenance headers, and C2PA markers.
    """
    with Image.open(image_path) as img:
        info = dict(img.info)
        format_name = img.format
        mode = img.mode
        size = img.size

        # Check for C2PA raw bytes in PNG / JPEG
        has_c2pa = False
        with open(image_path, "rb") as f:
            raw_bytes = f.read()
            if C2PA_BOX_UUID in raw_bytes or b"c2pa" in raw_bytes.lower() or b"jumbf" in raw_bytes.lower():
                has_c2pa = True

        exif_keys = []
        if hasattr(img, "_getexif") and img._getexif():
            exif_keys = list(img._getexif().keys())

        # Check for AI generator signature strings
        ai_markers = []
        lower_bytes = raw_bytes.lower()
        for marker in [b"gemini", b"imagen", b"synthid", b"stable diffusion", b"midjourney", b"dall-e", b"openai"]:
            if marker in lower_bytes:
                ai_markers.append(marker.decode())

    return {
        "format": format_name,
        "dimensions": f"{size[0]}x{size[1]}",
        "color_mode": mode,
        "has_c2pa": has_c2pa,
        "metadata_keys": list(info.keys()),
        "exif_tags_count": len(exif_keys),
        "ai_signature_markers": ai_markers,
        "file_size_bytes": os.path.getsize(image_path),
    }


def generate_difference_heatmap(orig_path: str, cleaned_path: str, output_diff_path: str, amplify: float = 15.0) -> dict:
    """
    Computes amplified pixel-wise difference between original and cleaned image,
    visualizing the steganographic noise perturbation.
    """
    with Image.open(orig_path) as img1, Image.open(cleaned_path) as img2:
        arr1 = np.array(img1.convert("RGB"), dtype=np.float32)
        arr2 = np.array(img2.convert("RGB"), dtype=np.float32)

    # Ensure matching dimensions
    min_h = min(arr1.shape[0], arr2.shape[0])
    min_w = min(arr1.shape[1], arr2.shape[1])
    diff = np.abs(arr1[:min_h, :min_w] - arr2[:min_h, :min_w])

    # Compute mean and max perturbation
    mean_diff = float(np.mean(diff))
    max_diff = float(np.max(diff))

    # Amplify difference for visual inspection
    amplified = np.clip(diff * amplify, 0.0, 255.0).astype(np.uint8)

    diff_img = Image.fromarray(amplified)
    diff_img.save(output_diff_path)

    return {
        "diff_path": output_diff_path,
        "mean_pixel_delta": round(mean_diff, 4),
        "max_pixel_delta": round(max_diff, 4),
        "amplification_factor": amplify,
    }


def compute_fft_magnitude_spectrum(img_arr: np.ndarray) -> np.ndarray:
    """
    Computes shifted 2D Fast Fourier Transform magnitude spectrum in decibels (dB).
    Used to detect high-frequency periodic carriers indicative of watermarks.
    """
    if img_arr.ndim == 3:
        # Convert to grayscale luminance
        gray = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
    else:
        gray = img_arr.astype(np.float32)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    # Normalize to 0-255
    norm_spec = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.ptp(magnitude_spectrum) + 1e-7) * 255.0
    return norm_spec.astype(np.uint8)

