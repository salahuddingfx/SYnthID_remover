"""
Main Application for Hugging Face Spaces & Cloud Deployment.
Supports both Gradio (native HF Spaces) and standalone Web UI.
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image

from synthid_remover.core.pipeline import SynthIDPurifier
from synthid_remover.filters.resizer import split_image_grid, SOCIAL_PRESETS
from synthid_remover.core.analyzer import generate_difference_heatmap

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


def purify_gradio(image, mode, strength, upscale, social_preset, remove_badge, split_grid):
    if image is None:
        return None, None, [], "⚠️ Please upload an image first."

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "in.png")
        output_path = os.path.join(tmpdir, "out.png")
        diff_path = os.path.join(tmpdir, "diff.png")

        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(input_path)
        else:
            image.save(input_path)

        purifier = SynthIDPurifier(
            mode=mode,
            strength=float(strength),
            remove_badge=remove_badge,
            upscale=int(upscale),
            social_preset=social_preset if social_preset != "None" else None,
        )
        res = purifier.process_image(input_path, output_path, output_format="PNG")

        cleaned_pil = Image.open(output_path).copy()

        # Generate difference heatmap
        diff_pil = None
        if mode != "metadata_only":
            try:
                generate_difference_heatmap(input_path, output_path, diff_path, amplify=15.0)
                diff_pil = Image.open(diff_path).copy()
            except Exception:
                pass

        # Split grid tiles
        tiles = []
        if split_grid and split_grid != "None":
            if split_grid == "1x2 Panorama":
                tiles = split_image_grid(cleaned_pil, rows=1, cols=2)
            elif split_grid == "1x3 Panorama":
                tiles = split_image_grid(cleaned_pil, rows=1, cols=3)
            elif split_grid == "3x3 Grid":
                tiles = split_image_grid(cleaned_pil, rows=3, cols=3)

        metrics = res.get("metrics", {})
        report = (
            f"### 🛡️ Purification Metrics\n\n"
            f"- **PSNR Fidelity**: `{metrics.get('psnr_db', '--')} dB`\n"
            f"- **SSIM Score**: `{metrics.get('ssim', '--')}`\n"
            f"- **Quality Retention**: `{metrics.get('quality_preservation', '--')}`\n"
            f"- **C2PA / EXIF / XMP**: `100% Stripped`\n"
            f"- **Dimensions**: `{cleaned_pil.width} × {cleaned_pil.height}`\n"
        )

        return cleaned_pil, diff_pil, tiles, report


def build_gradio_app():
    with gr.Blocks(title="SynthID & AI Watermark Remover", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🛡️ SynthID & AI Watermark Remover\n"
            "**Remove invisible SynthID watermarks, C2PA manifests, and EXIF metadata from AI-generated images with near-lossless visual quality.**"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Upload AI Generated Image")

                with gr.Accordion("⚙️ Removal & Formatting Settings", open=True):
                    mode_radio = gr.Radio(
                        choices=["balanced", "deep", "fast", "metadata_only"],
                        value="balanced",
                        label="Purification Mode",
                        info="'balanced' uses 2D DCT + Bilateral + Grid Desync; 'deep' adds global frequency perturbation."
                    )
                    strength_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.35, step=0.05,
                        label="Perturbation Strength",
                        info="0.35 is optimal for PSNR > 41 dB"
                    )
                    upscale_dropdown = gr.Dropdown(
                        choices=["1", "2", "4"],
                        value="1",
                        label="Super-Resolution Upscaler",
                        info="Upscale 2x or 4x with edge-directed sharpening"
                    )
                    social_dropdown = gr.Dropdown(
                        choices=["None", "facebook_post", "facebook_square", "instagram_portrait", "instagram_story", "twitter_post", "youtube_thumb"],
                        value="None",
                        label="Social Media Sizer",
                        info="Formats image dimensions for Facebook, Instagram, TikTok, etc."
                    )
                    split_dropdown = gr.Dropdown(
                        choices=["None", "1x2 Panorama", "1x3 Panorama", "3x3 Grid"],
                        value="None",
                        label="Pixel Divider / Carousel Splitter",
                        info="Splits wide images into multi-photo carousel tiles"
                    )
                    badge_checkbox = gr.Checkbox(
                        value=False,
                        label="Remove Visible Badge (Inpaint Gemini sparkle stamp)"
                    )

                purify_btn = gr.Button("🚀 Purify Image", variant="primary", size="lg")

            with gr.Column(scale=1):
                metrics_md = gr.Markdown("### 🛡️ Status: Ready")
                output_clean = gr.Image(type="pil", label="Purified Image (Clean & Lossless)")

                with gr.Tab("Watermark Residual Heatmap"):
                    output_diff = gr.Image(type="pil", label="Amplified Watermark Noise (×15)")

                with gr.Tab("Carousel Tiles (Pixel Divider)"):
                    output_gallery = gr.Gallery(label="Split Tiles", columns=3, height="auto")

        purify_btn.click(
            fn=purify_gradio,
            inputs=[input_img, mode_radio, strength_slider, upscale_dropdown, social_dropdown, badge_checkbox, split_dropdown],
            outputs=[output_clean, output_diff, output_gallery, metrics_md]
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    if HAS_GRADIO:
        print(f"🚀 Launching Gradio UI on port {port}...")
        demo = build_gradio_app()
        demo.launch(server_name="0.0.0.0", server_port=port)
    else:
        print(f"🚀 Launching Standalone Web UI on port {port}...")
        from synthid_remover.web.server import start_server
        start_server(host="0.0.0.0", port=port)
