"""
SynthID Remover package.
"""

from synthid_remover.core.pipeline import SynthIDPurifier
from synthid_remover.metadata.cleaner import clean_image_metadata, clean_video_c2pa
from synthid_remover.core.metrics import calculate_psnr, calculate_ssim, evaluate_fidelity

__version__ = "1.0.0"
__all__ = [
    "SynthIDPurifier",
    "clean_image_metadata",
    "clean_video_c2pa",
    "calculate_psnr",
    "calculate_ssim",
    "evaluate_fidelity",
]
