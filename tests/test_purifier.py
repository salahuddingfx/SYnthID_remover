"""
Unit and integration tests for SynthID Remover.
"""

import os
import tempfile
import struct
import numpy as np
from PIL import Image

from synthid_remover.core.pipeline import SynthIDPurifier
from synthid_remover.metadata.cleaner import clean_video_c2pa, C2PA_BOX_UUID


def create_synthetic_watermarked_image(path: str, size=(256, 256)):
    """Creates a synthetic image with an imperceptible steganographic carrier signal."""
    # Base gradient image
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    xx, yy = np.meshgrid(x, y)
    base = (np.sin(xx * 5) * np.cos(yy * 5) * 127 + 128).astype(np.float32)
    rgb = np.stack([base, base * 0.8, base * 0.6], axis=-1)

    # Add synthetic high-frequency steganographic watermark pattern (like SynthID)
    u = np.linspace(0, 50, size[0])
    v = np.linspace(0, 50, size[1])
    uu, vv = np.meshgrid(u, v)
    watermark_carrier = (np.sin(uu * 2.5) * np.cos(vv * 2.5) * 4.0).astype(np.float32)
    watermarked = np.clip(rgb + watermark_carrier[:, :, None], 0, 255).astype(np.uint8)

    img = Image.fromarray(watermarked)
    # Save with custom metadata
    info = {"comment": "AI Generated Imagen3 with SynthID"}
    img.save(path, format="PNG")
    return watermarked


def test_purification_modes():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_img_path = os.path.join(tmpdir, "input.png")
        create_synthetic_watermarked_image(input_img_path)

        for mode in ["fast", "balanced", "deep", "metadata_only"]:
            out_path = os.path.join(tmpdir, f"out_{mode}.png")
            purifier = SynthIDPurifier(mode=mode, strength=0.35)
            res = purifier.process_image(input_img_path, out_path)

            assert os.path.exists(out_path), f"Output file missing for mode {mode}"
            assert res["mode"] == mode

            with Image.open(out_path) as out_img:
                assert out_img.size == (256, 256)

            metrics = res["metrics"]
            if mode != "metadata_only":
                assert metrics["psnr_db"] > 35.0, f"PSNR too low for mode {mode}: {metrics['psnr_db']}"
                assert metrics["ssim"] > 0.95, f"SSIM too low for mode {mode}: {metrics['ssim']}"
            print(f"✓ Mode '{mode}' passed! PSNR: {metrics['psnr_db']} dB, SSIM: {metrics['ssim']}")


def test_video_c2pa_atom_scrubber():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_mp4 = os.path.join(tmpdir, "test.mp4")
        cleaned_mp4 = os.path.join(tmpdir, "cleaned.mp4")

        # Create dummy MP4 atoms: ftyp atom + uuid atom with C2PA uuid
        ftyp_data = b"isom" + b"\x00\x00\x02\x00" + b"isomiso2"
        ftyp_box = struct.pack(">I", 8 + len(ftyp_data)) + b"ftyp" + ftyp_data

        uuid_payload = C2PA_BOX_UUID + b"c2pa_manifest_data_payload_here"
        uuid_box = struct.pack(">I", 8 + len(uuid_payload)) + b"uuid" + uuid_payload

        with open(fake_mp4, "wb") as f:
            f.write(ftyp_box + uuid_box)

        res = clean_video_c2pa(fake_mp4, cleaned_mp4)
        assert res["boxes_modified"] == 1, "Failed to modify C2PA uuid box"

        # Verify uuid was rewritten to 'free'
        with open(cleaned_mp4, "rb") as f:
            cleaned_data = f.read()

        assert b"uuid" not in cleaned_data
        assert b"free" in cleaned_data
        print("✓ Video C2PA box stripping passed!")


if __name__ == "__main__":
    test_purification_modes()
    test_video_c2pa_atom_scrubber()
    print("\n🎉 All tests passed successfully!")
