"""
Lightweight Web Interface for SynthID Remover.
Runs a zero-dependency local web app with drag-and-drop, interactive comparison, and live metrics.
"""

import os
import io
import json
import base64
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from PIL import Image

from synthid_remover.core.pipeline import SynthIDPurifier
from synthid_remover.core.analyzer import inspect_image_metadata, generate_difference_heatmap

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SynthID & AI Watermark Remover</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --brand: #38bdf8;
      --brand-hover: #0284c7;
      --text: #f8fafc;
      --muted: #94a3b8;
      --success: #4ade80;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }
    .container { max-width: 900px; width: 100%; }
    header { text-align: center; margin-bottom: 2rem; }
    h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text); }
    h1 span { color: var(--brand); }
    p.subtitle { color: var(--muted); font-size: 1.05rem; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.75rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .drop-zone {
      border: 2px dashed var(--border);
      border-radius: 12px;
      padding: 2.5rem 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(15, 23, 42, 0.4);
    }
    .drop-zone:hover, .drop-zone.dragover {
      border-color: var(--brand);
      background: rgba(56, 189, 248, 0.05);
    }
    .drop-zone p { color: var(--muted); margin-top: 0.5rem; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.25rem; }
    label { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
    select, input[type=range] {
      width: 100%; margin-top: 0.35rem; padding: 0.6rem; border-radius: 8px;
      background: #0f172a; border: 1px solid var(--border); color: var(--text); font-size: 0.95rem;
    }
    .btn {
      display: inline-flex; align-items: center; justify-content: center; width: 100%;
      background: var(--brand); color: #0f172a; font-weight: 700; font-size: 1.05rem;
      padding: 0.85rem; border-radius: 10px; border: none; cursor: pointer;
      margin-top: 1.5rem; transition: background 0.2s;
    }
    .btn:hover { background: var(--brand-hover); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .metrics-bar {
      display: flex; justify-content: space-around; margin-top: 1.5rem; padding: 1rem;
      background: #0f172a; border-radius: 10px; border: 1px solid var(--border);
    }
    .metric { text-align: center; }
    .metric .val { font-size: 1.3rem; font-weight: 700; color: var(--brand); }
    .metric .lbl { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }
    .preview-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem; }
    .preview-box { text-align: center; }
    .preview-box img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); margin-top: 0.5rem; }
    .status-badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }
    .status-badge.clean { background: rgba(74, 222, 128, 0.15); color: var(--success); }
    .download-btn {
      display: inline-block; background: var(--success); color: #0f172a; font-weight: 700;
      padding: 0.7rem 1.5rem; border-radius: 8px; text-decoration: none; margin-top: 1rem;
    }
    /* Comparison Slider */
    .slider-container {
      position: relative;
      width: 100%;
      max-width: 700px;
      margin: 1.5rem auto 0 auto;
      overflow: hidden;
      border-radius: 10px;
      border: 1px solid var(--border);
      user-select: none;
    }
    .slider-img {
      display: block;
      width: 100%;
      height: auto;
      pointer-events: none;
    }
    .slider-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 50%;
      height: 100%;
      overflow: hidden;
    }
    .slider-handle {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 3px;
      background: var(--brand);
      cursor: ew-resize;
    }
    .slider-handle::after {
      content: "⟨ ⟩";
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: var(--brand);
      color: #0f172a;
      font-weight: 900;
      font-size: 11px;
      border-radius: 50%;
      width: 28px;
      height: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    .view-toggle {
      display: flex;
      justify-content: center;
      gap: 0.5rem;
      margin-top: 1.5rem;
    }
    .view-btn {
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 0.4rem 1rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
    }
    .view-btn.active {
      background: var(--brand);
      color: #0f172a;
      border-color: var(--brand);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>SynthID <span>Remover</span></h1>
      <p class="subtitle">100% Local AI Watermark & C2PA Purifier with Zero Quality Loss</p>
    </header>

    <div class="card">
      <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--brand); margin: 0 auto 0.5rem;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
        <h3>Click or drag & drop AI image here</h3>
        <p id="fileLabel">Supports PNG, JPEG, WEBP</p>
      </div>
      <input type="file" id="fileInput" accept="image/*" style="display: none;">

      <div class="controls">
        <div>
          <label for="modeSelect">Purification Mode</label>
          <select id="modeSelect">
            <option value="balanced" selected>Balanced (2D DCT + Bilateral + Grid Desync)</option>
            <option value="deep">Deep / Aggressive (Global + Block DCT + Multi-band)</option>
            <option value="fast">Fast (Spatial Bilateral + Chroma Dither)</option>
            <option value="metadata_only">Metadata Only (C2PA & EXIF Purge)</option>
          </select>
        </div>
        <div>
          <label for="strengthRange">Strength: <span id="strengthVal">0.35</span></label>
          <input type="range" id="strengthRange" min="0.1" max="1.0" step="0.05" value="0.35" oninput="document.getElementById('strengthVal').innerText = this.value">
        </div>
      </div>

      <div class="controls" style="margin-top: 1rem;">
        <div>
          <label for="upscaleSelect">Resolution Upscaler</label>
          <select id="upscaleSelect">
            <option value="1" selected>Original Size (1x)</option>
            <option value="2">2x Super-Resolution (Crisp)</option>
            <option value="4">4x Ultra-Resolution (High Detail)</option>
          </select>
        </div>
        <div>
          <label for="socialSelect">Social Media Sizer</label>
          <select id="socialSelect">
            <option value="" selected>Original Aspect Ratio</option>
            <option value="facebook_post">Facebook Feed Post (1200x630)</option>
            <option value="facebook_square">Facebook / IG Square (1080x1080)</option>
            <option value="instagram_portrait">Instagram Portrait 4:5 (1080x1350)</option>
            <option value="instagram_story">Instagram Story / Reels / TikTok (1080x1920)</option>
            <option value="twitter_post">Twitter / X Post (1200x675)</option>
            <option value="youtube_thumb">YouTube Thumbnail (1280x720)</option>
          </select>
        </div>
      </div>

      <div class="controls" style="margin-top: 1rem;">
        <div>
          <label for="splitSelect">Pixel Divider / Carousel Splitter</label>
          <select id="splitSelect">
            <option value="" selected>No Split (Single Image)</option>
            <option value="1x2">1x2 Panorama (2-slide Carousel)</option>
            <option value="1x3">1x3 Panorama (3-slide Carousel)</option>
            <option value="3x3">3x3 Grid (9-tile Profile Feed)</option>
          </select>
        </div>
        <div style="display: flex; align-items: flex-end; padding-bottom: 0.25rem;">
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <input type="checkbox" id="badgeCheck" style="width: 18px; height: 18px; cursor: pointer;">
            <label for="badgeCheck" style="cursor: pointer; text-transform: none; font-size: 0.95rem; color: var(--text);">Remove Visible Badge (Inpaint Gemini sparkle)</label>
          </div>
        </div>
      </div>

      <button class="btn" id="processBtn" disabled onclick="processImage()">Purify Image</button>
    </div>

    <div class="card" id="resultCard" style="display: none;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <h2>Purification Results</h2>
        <span class="status-badge clean">Watermark Disrupted</span>
      </div>

      <div class="metrics-bar">
        <div class="metric"><div class="val" id="psnrVal">--</div><div class="lbl">PSNR Fidelity</div></div>
        <div class="metric"><div class="val" id="ssimVal">--</div><div class="lbl">SSIM Score</div></div>
        <div class="metric"><div class="val" id="c2paVal">Stripped</div><div class="lbl">C2PA / EXIF</div></div>
        <div class="metric"><div class="val" id="qualityVal">--</div><div class="lbl">Visual Retention</div></div>
      </div>

      <div class="view-toggle">
        <button class="view-btn active" id="btnSideBySide" onclick="switchView('sideBySide')">Side-by-Side</button>
        <button class="view-btn" id="btnSlider" onclick="switchView('slider')">Interactive Split Slider</button>
        <button class="view-btn" id="btnDiff" onclick="switchView('diff')">Watermark Residual Heatmap</button>
      </div>

      <div class="preview-grid" id="viewSideBySide">
        <div class="preview-box">
          <h4>Original Image</h4>
          <img id="origPreview">
        </div>
        <div class="preview-box">
          <h4>Purified (Clean)</h4>
          <img id="cleanPreview">
        </div>
      </div>

      <div id="viewSlider" style="display: none;">
        <div class="slider-container" id="sliderContainer">
          <img id="sliderClean" class="slider-img" alt="Cleaned">
          <div class="slider-overlay" id="sliderOverlay">
            <img id="sliderOrig" class="slider-img" alt="Original" style="position: absolute; top: 0; left: 0; max-width: none;">
          </div>
          <div class="slider-handle" id="sliderHandle"></div>
        </div>
        <p style="text-align: center; font-size: 0.8rem; color: var(--muted); margin-top: 0.5rem;">Drag the divider to compare Original (Left) vs Cleaned (Right)</p>
      </div>

      <div id="viewDiff" style="display: none; text-align: center;">
        <h4 style="margin-top: 1rem;">Amplified Steganographic Perturbation Residual (×15)</h4>
        <p style="font-size: 0.85rem; color: var(--muted); margin-bottom: 0.5rem;">Visualizes the neutralized SynthID carrier frequency noise.</p>
        <img id="diffPreview" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--border);">
      </div>

      <div id="gridTilesContainer" style="display: none; margin-top: 1.5rem;">
        <h3 style="text-align: center; margin-bottom: 0.75rem;">✂️ Pixel Divider / Carousel Tiles</h3>
        <div id="gridTilesList" style="display: flex; flex-wrap: wrap; gap: 0.75rem; justify-content: center;"></div>
        <div style="text-align: center; margin-top: 0.75rem;">
          <a id="downloadAllTilesBtn" class="download-btn" style="background: var(--brand); font-size: 0.9rem; padding: 0.5rem 1.2rem; cursor: pointer;">Download All Tiles (.ZIP)</a>
        </div>
      </div>

      <div style="text-align: center; margin-top: 1.5rem;">
        <a id="downloadLink" class="download-btn" download="cleaned_image.png">Download Clean Image</a>
      </div>
    </div>
  </div>

  <script>
    let selectedFile = null;
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault(); dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

    function handleFile(file) {
      selectedFile = file;
      document.getElementById('fileLabel').innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      processBtn.disabled = false;
      const reader = new FileReader();
      reader.onload = (e) => {
        document.getElementById('origPreview').src = e.target.result;
        document.getElementById('sliderOrig').src = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    function switchView(view) {
      ['sideBySide', 'slider', 'diff'].forEach(v => {
        const el = document.getElementById('view' + v.charAt(0).toUpperCase() + v.slice(1));
        const btn = document.getElementById('btn' + v.charAt(0).toUpperCase() + v.slice(1));
        if (el) el.style.display = (v === view ? 'block' : 'none');
        if (btn) btn.classList.toggle('active', v === view);
      });
      if (view === 'slider') initSlider();
    }

    function initSlider() {
      const container = document.getElementById('sliderContainer');
      const sliderOrig = document.getElementById('sliderOrig');
      const overlay = document.getElementById('sliderOverlay');
      const handle = document.getElementById('sliderHandle');
      if (!container || !sliderOrig) return;

      const w = container.offsetWidth;
      sliderOrig.style.width = w + 'px';

      let isDragging = false;
      function setSliderPos(x) {
        const rect = container.getBoundingClientRect();
        let pos = Math.max(0, Math.min(x - rect.left, rect.width));
        let pct = (pos / rect.width) * 100;
        overlay.style.width = pct + '%';
        handle.style.left = pct + '%';
      }

      container.onmousedown = (e) => { isDragging = true; setSliderPos(e.clientX); };
      window.onmousemove = (e) => { if (isDragging) setSliderPos(e.clientX); };
      window.onmouseup = () => { isDragging = false; };
      container.ontouchstart = (e) => { isDragging = true; setSliderPos(e.touches[0].clientX); };
      window.ontouchmove = (e) => { if (isDragging) setSliderPos(e.touches[0].clientX); };
      window.ontouchend = () => { isDragging = false; };
    }

    async function processImage() {
      if (!selectedFile) return;
      processBtn.disabled = true;
      processBtn.innerText = 'Purifying...';

      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('mode', document.getElementById('modeSelect').value);
      formData.append('strength', document.getElementById('strengthRange').value);
      formData.append('remove_badge', document.getElementById('badgeCheck').checked ? 'true' : 'false');
      formData.append('upscale', document.getElementById('upscaleSelect').value);
      formData.append('social_preset', document.getElementById('socialSelect').value);
      formData.append('split_grid', document.getElementById('splitSelect').value);

      try {
        const res = await fetch('/api/purify', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
          document.getElementById('resultCard').style.display = 'block';
          document.getElementById('cleanPreview').src = data.cleaned_data_url;
          document.getElementById('sliderClean').src = data.cleaned_data_url;
          if (data.diff_data_url) {
            document.getElementById('diffPreview').src = data.diff_data_url;
          }
          document.getElementById('downloadLink').href = data.cleaned_data_url;
          document.getElementById('downloadLink').download = 'cleaned_' + selectedFile.name;
          document.getElementById('psnrVal').innerText = data.metrics.psnr_db + ' dB';
          document.getElementById('ssimVal').innerText = data.metrics.ssim;
          document.getElementById('qualityVal').innerText = data.metrics.quality_preservation;

          // Render grid tiles if generated
          const gridBox = document.getElementById('gridTilesContainer');
          const gridList = document.getElementById('gridTilesList');
          const zipBtn = document.getElementById('downloadAllTilesBtn');
          gridList.innerHTML = '';
          if (data.tiles && data.tiles.length > 0) {
            gridBox.style.display = 'block';
            data.tiles.forEach((t, i) => {
              const item = document.createElement('div');
              item.style.textAlign = 'center';
              item.innerHTML = `
                <img src="${t.data_url}" style="width: 140px; height: 140px; object-fit: cover; border-radius: 8px; border: 1px solid var(--border);">
                <br>
                <a href="${t.data_url}" download="tile_${i+1}.png" style="font-size: 0.8rem; color: var(--brand); text-decoration: underline; font-weight: 600;">Tile #${i+1}</a>
              `;
              gridList.appendChild(item);
            });
            if (data.zip_data_url) {
              zipBtn.href = data.zip_data_url;
              zipBtn.download = 'carousel_tiles.zip';
              zipBtn.style.display = 'inline-block';
            } else {
              zipBtn.style.display = 'none';
            }
          } else {
            gridBox.style.display = 'none';
          }
        } else {
          alert('Error: ' + data.error);
        }
      } catch (err) {
        alert('Processing failed: ' + err);
      } finally {
        processBtn.disabled = false;
        processBtn.innerText = 'Purify Image';
      }
    }
  </script>
</body>
</html>
"""


class SynthIDWebHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/purify":
                self.handle_purify()
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as err:
            import traceback
            traceback.print_exc()
            self.send_json({"success": False, "error": str(err)}, 500)

    def handle_purify(self):
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_json({"success": False, "error": "Invalid Content-Type"}, 400)
            return

        # Read multipart body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        boundary = content_type.split("boundary=")[-1].strip().encode()

        # Parse multipart parts
        parts = body.split(b"--" + boundary)
        file_bytes = None
        mode = "balanced"
        strength = 0.35
        remove_badge = False
        upscale = 1
        social_preset = None
        split_grid = None

        for part in parts:
            if b'name="image"' in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end != -1:
                    file_bytes = part[header_end + 4 :].rstrip(b"\r\n")
            elif b'name="mode"' in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end != -1:
                    mode = part[header_end + 4 :].rstrip(b"\r\n").decode("utf-8", errors="ignore")
            elif b'name="strength"' in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end != -1:
                    try:
                        strength = float(part[header_end + 4 :].rstrip(b"\r\n"))
                    except ValueError:
                        pass
            elif b'name="remove_badge"' in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end != -1:
                    val = part[header_end + 4 :].rstrip(b"\r\n").decode("utf-8", errors="ignore")
                    remove_badge = (val == "true")
            elif b'name="upscale"' in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end != -1:
                    try:
                        upscale = int(part[header_end + 4 :].rstrip(b"\r\n"))
                    except ValueError:
                        pass
            elif b'name="social_preset"' in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end != -1:
                    val = part[header_end + 4 :].rstrip(b"\r\n").decode("utf-8", errors="ignore").strip()
                    if val:
                        social_preset = val
            elif b'name="split_grid"' in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end != -1:
                    val = part[header_end + 4 :].rstrip(b"\r\n").decode("utf-8", errors="ignore").strip()
                    if val:
                        split_grid = val

        if not file_bytes:
            self.send_json({"success": False, "error": "No image uploaded"}, 400)
            return

        # Process image in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.png")
            output_file = os.path.join(tmpdir, "output.png")
            diff_file = os.path.join(tmpdir, "diff.png")

            with open(input_file, "wb") as f:
                f.write(file_bytes)

            purifier = SynthIDPurifier(
                mode=mode,
                strength=strength,
                remove_badge=remove_badge,
                upscale=upscale,
                social_preset=social_preset,
            )
            res = purifier.process_image(input_file, output_file, output_format="PNG")

            with open(output_file, "rb") as f:
                cleaned_bytes = f.read()

            b64_img = base64.b64encode(cleaned_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{b64_img}"

            diff_data_url = None
            if mode != "metadata_only":
                try:
                    generate_difference_heatmap(input_file, output_file, diff_file, amplify=15.0)
                    with open(diff_file, "rb") as df:
                        diff_b64 = base64.b64encode(df.read()).decode("utf-8")
                        diff_data_url = f"data:image/png;base64,{diff_b64}"
                except Exception:
                    pass

            # Grid Tiles
            tiles_payload = []
            zip_data_url = None
            if split_grid:
                import zipfile
                from synthid_remover.filters.resizer import split_image_grid
                with Image.open(output_file) as out_img:
                    if split_grid == "1x2":
                        tile_imgs = split_image_grid(out_img, rows=1, cols=2)
                    elif split_grid == "1x3":
                        tile_imgs = split_image_grid(out_img, rows=1, cols=3)
                    elif split_grid == "3x3":
                        tile_imgs = split_image_grid(out_img, rows=3, cols=3)
                    else:
                        tile_imgs = []

                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for t_idx, t_img in enumerate(tile_imgs):
                            buf = io.BytesIO()
                            t_img.save(buf, format="PNG")
                            img_bytes = buf.getvalue()
                            t_b64 = base64.b64encode(img_bytes).decode("utf-8")
                            tiles_payload.append({
                                "tile": t_idx + 1,
                                "data_url": f"data:image/png;base64,{t_b64}"
                            })
                            zf.writestr(f"carousel_tile_{t_idx+1}.png", img_bytes)

                    zip_b64 = base64.b64encode(zip_buf.getvalue()).decode("utf-8")
                    zip_data_url = f"data:application/zip;base64,{zip_b64}"

            self.send_json({
                "success": True,
                "cleaned_data_url": data_url,
                "diff_data_url": diff_data_url,
                "tiles": tiles_payload,
                "zip_data_url": zip_data_url,
                "metrics": res.get("metrics", {}),
                "mode": mode,
            })

    def send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def start_server(host: str = "0.0.0.0", port: int = None):
    if port is None:
        port = int(os.environ.get("PORT", 8080))
    server = HTTPServer((host, port), SynthIDWebHandler)
    print(f"🚀 SynthID Remover Web UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()


if __name__ == "__main__":
    start_server()

