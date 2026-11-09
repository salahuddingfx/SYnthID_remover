"""
Metadata and Provenance Scrubber.
Strips C2PA manifests, EXIF, XMP, IPTC, and proprietary AI metadata blocks from images and videos.
"""

import io
import struct
from PIL import Image

C2PA_BOX_UUID = b"\xd8\xf1\x32\x42\x4c\x69\x11\xec\xa9\x6c\x1f\x5a\xaa\xa6\x97\x0a"


def clean_image_metadata(input_path: str, output_path: str, output_format: str = None) -> dict:
    """
    Strips all metadata (EXIF, XMP, IPTC, ICC, PNG text chunks) from an image
    and writes out clean raw pixel data.
    """
    with Image.open(input_path) as img:
        orig_format = img.format or "PNG"
        target_format = (output_format or orig_format).upper()
        if target_format == "JPG":
            target_format = "JPEG"

        # Convert to RGB or RGBA discarding any palette or color profile ties
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            clean_img = img.convert("RGBA")
        else:
            clean_img = img.convert("RGB")

        # Create brand-new PIL image copying only raw pixel buffer
        pixels = clean_img.copy()

        save_kwargs = {}
        if target_format == "JPEG":
            save_kwargs["quality"] = 98
            save_kwargs["subsampling"] = 0
            save_kwargs["optimize"] = True
        elif target_format == "PNG":
            save_kwargs["optimize"] = True
            save_kwargs["compress_level"] = 6
        elif target_format == "WEBP":
            save_kwargs["quality"] = 95
            save_kwargs["lossless"] = False

        pixels.save(output_path, format=target_format, **save_kwargs)

    return {
        "format": target_format,
        "width": pixels.width,
        "height": pixels.height,
        "metadata_stripped": True,
    }


def clean_video_c2pa(input_path: str, output_path: str) -> dict:
    """
    Cleans C2PA boxes from MP4/MOV container files by replacing C2PA uuid boxes
    with harmless 'free' atoms (matching the technique used by browser tools).
    """
    with open(input_path, "rb") as f:
        data = bytearray(f.read())

    offset = 0
    total_len = len(data)
    boxes_modified = 0

    while offset + 8 <= total_len:
        box_size = struct.unpack(">I", data[offset : offset + 4])[0]
        box_type = data[offset + 4 : offset + 8]

        header_size = 8
        if box_size == 1:  # 64-bit large size
            if offset + 16 > total_len:
                break
            box_size = struct.unpack(">Q", data[offset + 8 : offset + 16])[0]
            header_size = 16
        elif box_size == 0:  # Box extends to end of file
            box_size = total_len - offset

        if box_size < header_size:
            break

        # Check if box is a UUID box with C2PA UUID signature
        if box_type == b"uuid":
            uuid_offset = offset + header_size
            if uuid_offset + 16 <= total_len:
                uuid = data[uuid_offset : uuid_offset + 16]
                if uuid == C2PA_BOX_UUID or b"c2pa" in data[uuid_offset : offset + min(box_size, 128)]:
                    # Overwrite box type to 'free' (harmless padding box)
                    data[offset + 4 : offset + 8] = b"free"
                    boxes_modified += 1

        offset += box_size

    with open(output_path, "wb") as f:
        f.write(data)

    return {
        "boxes_modified": boxes_modified,
        "file_size": len(data),
    }
