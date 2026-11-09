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
        epilog="Examples:\n"
               "  ./synthid-remover -i input.png -o clean.png\n"
               "  ./synthid-remover -i ./images_dir/ -o ./cleaned_dir/\n"
               "  ./synthid-remover -i photo.jpg -m deep -s 0.45\n"
               "  ./synthid-remover --inspect image.png\n"
               "  ./synthid-remover --serve --port 8080\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", help="Path to input image/video file or directory."
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
        "--remove-badge",
        action="store_true",
        help="Detect and inpaint visible corner watermarks (e.g. Gemini sparkle badge).",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        choices=[2, 4],
        help="Upscale image resolution by 2x or 4x using high-order perceptual super-resolution.",
    )
    parser.add_argument(
        "--preset",
        choices=["facebook_post", "facebook_square", "instagram_post", "instagram_portrait", "instagram_story", "twitter_post", "youtube_thumb"],
        help="Format image to standard social media dimensions (e.g. 'facebook_post' 1200x630, 'instagram_story' 1080x1920).",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["pad_blur", "crop", "fit"],
        default="pad_blur",
        help="Fitting style for social media presets: 'pad_blur' (smart blurred background), 'crop' (center-crop), or 'fit'.",
    )
    parser.add_argument(
        "--split-grid",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLS"),
        help="Pixel divider: splits the purified image into a grid of ROWS x COLS tiles (e.g. 1 3 for Instagram panorama carousel).",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["png", "jpg", "jpeg", "webp"],
        help="Force output image format.",
    )
    parser.add_argument(
        "--diff-map",
        help="Save an amplified visual difference heatmap comparing original vs purified image.",
    )
    parser.add_argument(
        "--inspect",
        metavar="FILE",
        help="Scan and display embedded C2PA manifests, EXIF tags, and AI model signatures.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the local interactive Web UI (drag-and-drop, comparison slider, live metrics).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the local Web UI (default: 8080).",
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

    # Launch Web UI
    if args.serve:
        from synthid_remover.web.server import start_server
        start_server(port=args.port)
        return

    # Inspect image metadata / C2PA
    if args.inspect:
        from synthid_remover.core.analyzer import inspect_image_metadata
        target = os.path.abspath(args.inspect)
        if not os.path.exists(target):
            print(f"Error: Target file '{args.inspect}' does not exist.", file=sys.stderr)
            sys.exit(1)
        data = inspect_image_metadata(target)
        print(f"🔍 Inspection Report for: {target}")
        print(f"   Format: {data['format']} | Mode: {data['color_mode']} | Dimensions: {data['dimensions']}")
        print(f"   C2PA Credentials Found: {'⚠️ YES' if data['has_c2pa'] else '✓ None detected'}")
        print(f"   AI Generator Signatures: {', '.join(data['ai_signature_markers']) if data['ai_signature_markers'] else 'None'}")
        print(f"   Metadata Keys: {data['metadata_keys']}")
        print(f"   EXIF Tags: {data['exif_tags_count']}")
        return

    # If no input provided, print friendly help
    if not args.input:
        parser.print_help()
        print("\n💡 Quick Start:")
        print("  Run with an input file:  ./synthid-remover -i your_image.png")
        print("  Launch local Web UI:     ./synthid-remover --serve\n")
        sys.exit(0)

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    purifier = SynthIDPurifier(
        mode=args.mode,
        strength=args.strength,
        desync_grid=not args.no_desync,
        remove_badge=args.remove_badge,
        upscale=args.upscale or 1,
        social_preset=args.preset,
        resize_mode=args.resize_mode,
    )

    if os.path.isfile(input_path):
        if args.output:
            out_path = os.path.abspath(args.output)
        else:
            p = Path(input_path)
            ext = f".{args.format}" if args.format else p.suffix
            out_path = str(p.parent / f"cleaned_{p.stem}{ext}")

        process_single_file(input_path, out_path, purifier, args.format, args.quiet)

        if args.split_grid:
            from PIL import Image
            from synthid_remover.filters.resizer import split_image_grid
            rows, cols = args.split_grid
            with Image.open(out_path) as out_img:
                tiles = split_image_grid(out_img, rows=rows, cols=cols)
            p = Path(out_path)
            for idx, tile in enumerate(tiles):
                tile_path = str(p.parent / f"{p.stem}_tile_{idx+1}{p.suffix}")
                tile.save(tile_path)
                if not args.quiet:
                    print(f"   ✓ Grid tile {idx+1}/{len(tiles)} saved to: {tile_path}")

        if args.diff_map:
            from synthid_remover.core.analyzer import generate_difference_heatmap
            diff_res = generate_difference_heatmap(input_path, out_path, os.path.abspath(args.diff_map))
            if not args.quiet:
                print(f"   ✓ Difference heatmap saved to: {diff_res['diff_path']} (mean delta: {diff_res['mean_pixel_delta']})")

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

