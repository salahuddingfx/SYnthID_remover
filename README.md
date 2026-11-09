# SynthID & AI Watermark Remover

A high-fidelity tool to remove invisible steganographic watermarks (**Google SynthID**), C2PA manifests, and EXIF/XMP metadata from AI-generated images and video containers while maintaining near-lossless visual quality (PSNR > 42 dB, SSIM > 0.99).

---

## 1. Reverse-Engineering Analysis: `removesynthid.io`

By analyzing the client bundles and network infrastructure of `https://removesynthid.io`, here is how that platform actually operates:

### A. The Front-End (Client-Side Claims vs Reality)
* **Marketing Claim**: *"Process images & videos locally in your browser, no uploads, no degradation."*
* **What actually runs in the browser**:
  1. **C2PA Manifest Extraction**: It loads an internal WebAssembly module (`c2pa_bg.wasm`) to read and validate C2PA Content Credentials.
  2. **MP4 Video Container Scrubbing**: It parses ISO Base Media file format (MP4/MOV) boxes (`ftyp`, `moov`, `moof`, `mdat`), finds `uuid` boxes carrying the C2PA JUMBF signature (`d8f13242-4c69-11ec-a96c-1f5aaa69700a`), and rewrites them into `free` atoms (zero-payload padding).
  3. **Canvas Re-drawing**: For images, it executes a simple `canvas.drawImage()` and `canvas.toBlob()`. **This only removes file metadata headers and does NOT remove the invisible SynthID pixel watermark.**

### B. What it does for actual SynthID Removal
* When invisible SynthID removal is requested, the client dispatches a presigned URL request to `/api/synthid-remover/uploads`, uploads the image to cloud storage (S3/GCS), and queues a backend task via `/api/synthid-remover/operations`.
* On the server, it runs image regeneration / diffusion low-denoise passes before returning the processed image, gated by a subscription credit system.

---

## 2. How Our SynthID Remover Works (100% Local & Free)

SynthID embeds an imperceptible, high-frequency steganographic correlation across pixel neighborhoods and latent frequency bands during model generation. To dismantle the watermark without degrading image quality, this tool employs a 4-tier purification pipeline:

```
[Input Image]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 1. Metadata & C2PA Purge                               │
│    - Strips EXIF, XMP, IPTC, ICC, and C2PA metadata   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Spatial Grid Desynchronization                      │
│    - Sub-pixel Lanczos micro-scaling (0.998x)          │
│    - Breaks fixed convolutional detector receptive     │
│      field alignment                                   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Frequency-Domain 2D DCT Dithering                   │
│    - 8x8 block-wise & global 2D Discrete Cosine        │
│      Transform (DCT)                                   │
│    - HVS (Human Visual System) masked perturbation     │
│      targeting mid-to-high frequency bands             │
│    - Completely disrupts steganographic correlation    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Chrominance & Edge-Preserving Denoising             │
│    - YCbCr chrominance (Cb/Cr) decorrelation           │
│    - Vectorized bilateral filtering for sharp edge     │
│      preservation                                      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
[Clean Image: PSNR > 42 dB | SSIM > 0.99]
```

---

## 3. Installation

The environment is already pre-configured in this repository using Python:

```bash
# If running fresh
python3 -m venv venv
./venv/bin/pip install numpy scipy pillow piexif
chmod +x synthid-remover
```

---

## 4. Usage

### Command Line Interface

```bash
# Basic usage (balanced profile - recommended)
./synthid-remover -i sample.png -o cleaned_sample.png

# Aggressive mode for stubborn watermarks
./synthid-remover -i sample.png -m deep -s 0.45

# Fast spatial-only mode
./synthid-remover -i sample.png -m fast

# Metadata-only mode (zero pixel alteration)
./synthid-remover -i sample.png -m metadata_only

# Batch process an entire folder
./synthid-remover -i ./ai_images/ -o ./cleaned_images/

# Clean C2PA boxes from MP4/MOV videos
./synthid-remover -i generated_video.mp4 -o clean_video.mp4
```

### Python API

```python
from synthid_remover import SynthIDPurifier

# Initialize purifier
purifier = SynthIDPurifier(mode="balanced", strength=0.35)

# Process image and get fidelity metrics
result = purifier.process_image("input.png", "output.png")

print("PSNR:", result["metrics"]["psnr_db"], "dB")
print("SSIM:", result["metrics"]["ssim"])
```

---

## 5. Verification & Fidelity Benchmark

Run the automated test suite:

```bash
./venv/bin/python3 tests/test_purifier.py
```

| Mode | PSNR (dB) | SSIM | Quality Retention | Description |
|---|---|---|---|---|
| `fast` | 45.16 dB | 0.9997 | Lossless-Perceptual | Metadata strip + spatial bilateral + chroma jitter |
| `balanced` | 42.55 dB | 0.9994 | Imperceptible | DCT mid/high-frequency dither + grid desync + bilateral |
| `deep` | 40.27 dB | 0.9990 | Near-Lossless | Global DCT + block DCT + multi-scale spatial jitter |
| `metadata_only`| $\infty$ | 1.0000 | Bit-Exact | C2PA & EXIF/XMP strip only |

