"""
SynthID Image Purification Pipeline.
Orchestrates multi-layer defense against AI watermarks and steganographic provenance markers.
"""

import os
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image

from synthid_remover.metadata.cleaner import clean_image_metadata
from synthid_remover.filters.dct_filter import purify_dct_block, purify_dct_global
from synthid_remover.filters.spatial_purifier import desynchronize_grid, edge_preserving_denoise, chroma_decorrelation
from synthid_remover.filters.inpainter import remove_visible_badge
from synthid_remover.core.metrics import evaluate_fidelity


class SynthIDPurifier:
    """
    Main engine for removing SynthID and AI steganographic watermarks from images.
    """

    def __init__(self, mode: str = "balanced", strength: float = 0.35, desync_grid: bool = True, remove_badge: bool = False):
        """
        Args:
            mode: 'fast', 'balanced', 'deep', or 'metadata_only'
            strength: Perturbation strength for frequency/spatial filters (0.1 to 1.0)
            desync_grid: Whether to apply subpixel Lanczos desynchronization
            remove_badge: Whether to detect and inpaint visible corner badges (e.g. Gemini sparkle)
        """
        self.mode = mode.lower()
        self.strength = float(strength)
        self.desync_grid = desync_grid
        self.remove_badge = remove_badge

    def process_image(self, input_path: str, output_path: str, output_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the configured purification pipeline on an image file.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Metadata-only mode
        if self.mode == "metadata_only":
            meta_res = clean_image_metadata(input_path, output_path, output_format)
            return {
                "input": input_path,
                "output": output_path,
                "mode": self.mode,
                "metadata": meta_res,
                "metrics": {"psnr_db": "inf", "ssim": 1.0, "quality_preservation": "Lossless"},
            }

        # Load image
        with Image.open(input_path) as orig_pil:
            if self.remove_badge:
                orig_pil = remove_visible_badge(orig_pil, corner="bottom_right")
            # Normalize to RGB or RGBA
            has_alpha = orig_pil.mode in ("RGBA", "LA") or (orig_pil.mode == "P" and "transparency" in orig_pil.info)
            working_pil = orig_pil.convert("RGBA" if has_alpha else "RGB")
            orig_rgb = np.array(orig_pil.convert("RGB"), dtype=np.uint8)

        # Separate alpha channel if present
        working_arr = np.array(working_pil)
        if has_alpha:
            rgb_arr = working_arr[:, :, :3]
            alpha_channel = working_arr[:, :, 3]
        else:
            rgb_arr = working_arr
            alpha_channel = None

        # Step 1: Spatial Grid Desynchronization (disrupts fixed convolutional reception fields)
        if self.desync_grid and self.mode in ("balanced", "deep"):
            temp_img = Image.fromarray(rgb_arr)
            temp_img = desynchronize_grid(temp_img, scale_factor=0.998)
            rgb_arr = np.array(temp_img)

        # Step 2: Frequency-domain DCT Dithering
        if self.mode == "deep":
            # Global DCT followed by localized block DCT
            rgb_arr = purify_dct_global(rgb_arr, strength=self.strength * 0.7)
            rgb_arr = purify_dct_block(rgb_arr, strength=self.strength, seed=42)
        elif self.mode == "balanced":
            # Localized 8x8 block DCT with HVS quantization mask
            rgb_arr = purify_dct_block(rgb_arr, strength=self.strength, seed=42)

        # Step 3: Chrominance Decorrelation (SynthID frequently embeds in color difference channels)
        if self.mode in ("fast", "balanced", "deep"):
            jitter = 1.0 if self.mode == "fast" else (1.5 * self.strength)
            rgb_arr = chroma_decorrelation(rgb_arr, jitter_strength=jitter)

        # Step 4: Edge-Preserving Denoising (removes residual micro-steganographic perturbations)
        if self.mode in ("balanced", "deep"):
            rgb_arr = edge_preserving_denoise(
                rgb_arr,
                diameter=5,
                sigma_color=10.0 * self.strength,
                sigma_space=2.5,
            )

        # Recombine alpha channel if needed
        if has_alpha and alpha_channel is not None:
            final_arr = np.dstack([rgb_arr, alpha_channel])
            final_pil = Image.fromarray(final_arr, mode="RGBA")
        else:
            final_arr = rgb_arr
            final_pil = Image.fromarray(final_arr, mode="RGB")

        # Save output without any metadata chunks
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        ext = (output_format or os.path.splitext(output_path)[1].lstrip(".")).upper()
        if ext in ("JPG", "JPEG"):
            final_pil.convert("RGB").save(output_path, format="JPEG", quality=98, subsampling=0, optimize=True)
        elif ext == "WEBP":
            final_pil.save(output_path, format="WEBP", quality=95)
        else:
            final_pil.save(output_path, format="PNG", optimize=True)

        # Calculate mathematical fidelity
        cleaned_rgb = np.array(final_pil.convert("RGB"), dtype=np.uint8)
        fidelity = evaluate_fidelity(orig_rgb, cleaned_rgb)

        return {
            "input": input_path,
            "output": output_path,
            "mode": self.mode,
            "strength": self.strength,
            "dimensions": f"{final_pil.width}x{final_pil.height}",
            "metrics": fidelity,
        }

