"""
Social Media Sizer, Aspect Ratio Formatter, and Grid Pixel Divider.
Formats images for Facebook, Instagram, Twitter/X, TikTok, and YouTube,
with smart blurred-background padding and multi-panel grid splitting.
"""

from typing import Tuple, List, Dict
from PIL import Image, ImageFilter, ImageOps

SOCIAL_PRESETS = {
    "facebook_post": {"name": "Facebook Feed Post (1200x630)", "width": 1200, "height": 630},
    "facebook_square": {"name": "Facebook Square (1080x1080)", "width": 1080, "height": 1080},
    "instagram_post": {"name": "Instagram Square (1080x1080)", "width": 1080, "height": 1080},
    "instagram_portrait": {"name": "Instagram Portrait 4:5 (1080x1350)", "width": 1080, "height": 1350},
    "instagram_story": {"name": "Instagram Story / Reels / TikTok (1080x1920)", "width": 1080, "height": 1920},
    "twitter_post": {"name": "Twitter / X Post (1200x675)", "width": 1200, "height": 675},
    "youtube_thumb": {"name": "YouTube Thumbnail (1280x720)", "width": 1280, "height": 720},
}


def resize_to_social(
    img: Image.Image,
    target_width: int,
    target_height: int,
    mode: str = "pad_blur"
) -> Image.Image:
    """
    Resizes image to social media target dimensions.
    
    Modes:
      - 'pad_blur': Centers image over an aesthetically blurred & dimmed background of itself.
      - 'crop': Smart center-crop to fill target dimensions without stretching.
      - 'fit': Fits inside target canvas with subtle dark border padding.
    """
    orig_w, orig_h = img.size
    target_aspect = target_width / target_height
    orig_aspect = orig_w / orig_h

    if mode == "crop":
        # Scale so image covers the target entirely, then center crop
        scale = max(target_width / orig_w, target_height / orig_h)
        scaled_w = int(round(orig_w * scale))
        scaled_h = int(round(orig_h * scale))
        scaled_img = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        left = (scaled_w - target_width) // 2
        top = (scaled_h - target_height) // 2
        return scaled_img.crop((left, top, left + target_width, top + target_height))

    elif mode == "pad_blur":
        # Create blurred background that covers target canvas
        bg_scale = max(target_width / orig_w, target_height / orig_h)
        bg_w = int(round(orig_w * bg_scale))
        bg_h = int(round(orig_h * bg_scale))
        bg = img.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
        
        bg_left = (bg_w - target_width) // 2
        bg_top = (bg_h - target_height) // 2
        bg_cropped = bg.crop((bg_left, bg_top, bg_left + target_width, bg_top + target_height))
        # Heavy gaussian blur + slight dimming for background
        bg_blurred = bg_cropped.filter(ImageFilter.GaussianBlur(radius=30))
        # Darken background slightly to emphasize foreground
        bg_dimmed = Image.blend(bg_blurred, Image.new("RGB", (target_width, target_height), (15, 23, 42)), 0.35)

        # Scale foreground to fit inside target canvas
        fg_scale = min(target_width / orig_w, target_height / orig_h)
        fg_w = int(round(orig_w * fg_scale))
        fg_h = int(round(orig_h * fg_scale))
        fg_scaled = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

        fg_x = (target_width - fg_w) // 2
        fg_y = (target_height - fg_h) // 2

        # Paste foreground onto blurred background
        bg_dimmed.paste(fg_scaled, (fg_x, fg_y))
        return bg_dimmed

    else:  # 'fit'
        canvas = Image.new("RGB", (target_width, target_height), (15, 23, 42))
        scale = min(target_width / orig_w, target_height / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x = (target_width - new_w) // 2
        y = (target_height - new_h) // 2
        canvas.paste(scaled_img, (x, y))
        return canvas


def split_image_grid(img: Image.Image, rows: int = 1, cols: int = 3) -> List[Image.Image]:
    """
    Pixel Divider / Grid Splitter.
    Splits an image into rows x cols grid tiles (e.g. 1x3 carousel or 3x3 Instagram grid).
    """
    w, h = img.size
    tile_w = w // cols
    tile_h = h // rows

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            top = r * tile_h
            right = left + tile_w
            bottom = top + tile_h
            tile = img.crop((left, top, right, bottom))
            tiles.append(tile)

    return tiles

