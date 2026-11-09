"""
Command-Line Interface for SynthID Remover.
"""

import os
import sys
import argparse
import glob
from pathlib import Path

from synthid_remover.core.pipeline import SynthIDPurifier
from synthid_remover.metadata.cleaner import clean_video_c2pa

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".m4v"}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="synthid-remover",
        description="SynthID & AI Watermark Remover: Strips C2PA, EXIF, and invisible diffusion steganographic watermarks while preserving visual fidelity.",
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to input image/video file or directory."
    )
    parser.add_argument(
        "-o", "--output", help="Path to output file or directory (defaults to cleaned_<filename>)."
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["fast", "balanced", "deep", "metadata_only"],
        default="balanced",
        help="Purification mode. 'balanced' (default) uses DCT + bilateral + grid desync; 'deep' adds global frequency dithering; 'fast' uses spatial/chroma only; 'metadata_only' strips metadata without altering pixels.",
    )
    parser.add_argument(
        "-s", "--strength",
        type=float,
        default=0.35,
        help="Perturbation strength (0.1 to 1.0, default: 0.35).",
    )
    parser.add_argument(
        "--no-desync",
        action="store_true",
        help="Disable sub-pixel Lanczos grid desynchronization.",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["png", "jpg", "jpeg", "webp"],
        help="Force output image format.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed logs and print only file paths.",
    )
    return parser


def process_single_file(file_path: str, out_path: str, purifier: SynthIDPurifier, output_format: str, quiet: bool):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in SUPPORTED_VIDEO_EXTS:
        if not quiet:
            print(f"🎬 Processing video C2PA container: {file_path}")
        res = clean_video_c2pa(file_path, out_path)
        if not quiet:
            print(f"   ✓ Overwrote {res['boxes_modified']} C2PA manifest boxes with 'free' atoms.")
            print(f"   ✓ Saved cleaned video to: {out_path}")
        return

    if ext not in SUPPORTED_IMAGE_EXTS:
        if not quiet:
            print(f"⚠️  Skipping unsupported file: {file_path}")
        return

    if not quiet:
        print(f"🖼️  Purifying image: {file_path} (mode: {purifier.mode}, strength: {purifier.strength})")

    res = purifier.process_image(file_path, out_path, output_format=output_format)

    if not quiet:
        metrics = res.get("metrics", {})
        psnr_val = metrics.get("psnr_db")
        ssim_val = metrics.get("ssim")
        print(f"   ✓ Output saved to: {out_path}")
        print(f"   ✓ Metrics: PSNR: {psnr_val} dB | SSIM: {ssim_val} | Fidelity: {metrics.get('quality_preservation')}")
    else:
        print(out_path)


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    purifier = SynthIDPurifier(
        mode=args.mode,
        strength=args.strength,
        desync_grid=not args.no_desync,
    )

    if os.path.isfile(input_path):
        if args.output:
            out_path = os.path.abspath(args.output)
        else:
            p = Path(input_path)
            ext = f".{args.format}" if args.format else p.suffix
            out_path = str(p.parent / f"cleaned_{p.stem}{ext}")

        process_single_file(input_path, out_path, purifier, args.format, args.quiet)

    elif os.path.isdir(input_path):
        out_dir = os.path.abspath(args.output) if args.output else os.path.join(input_path, "cleaned")
        os.makedirs(out_dir, exist_ok=True)

        files = [
            f for f in sorted(glob.glob(os.path.join(input_path, "*")))
            if os.path.isfile(f) and Path(f).suffix.lower() in (SUPPORTED_IMAGE_EXTS | SUPPORTED_VIDEO_EXTS)
        ]

        if not files:
            print(f"No supported image/video files found in '{input_path}'.")
            return

        print(f"🚀 Batch processing {len(files)} files in '{input_path}' -> '{out_dir}'\n")
        for f in files:
            p = Path(f)
            ext = f".{args.format}" if args.format else p.suffix
            dest = os.path.join(out_dir, f"{p.stem}_cleaned{ext}")
            process_single_file(f, dest, purifier, args.format, args.quiet)
            print()

        print("✨ All files processed successfully.")


if __name__ == "__main__":
    main()
