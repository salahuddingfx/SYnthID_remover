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

      <div class="preview-grid">
        <div class="preview-box">
          <h4>Original Image</h4>
          <img id="origPreview">
        </div>
        <div class="preview-box">
          <h4>Purified (Clean)</h4>
          <img id="cleanPreview">
          <br>
          <a id="downloadLink" class="download-btn" download="cleaned_image.png">Download Clean Image</a>
        </div>
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
      reader.onload = (e) => { document.getElementById('origPreview').src = e.target.result; };
      reader.readAsDataURL(file);
    }

    async function processImage() {
      if (!selectedFile) return;
      processBtn.disabled = true;
      processBtn.innerText = 'Purifying...';

      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('mode', document.getElementById('modeSelect').value);
      formData.append('strength', document.getElementById('strengthRange').value);

      try {
        const res = await fetch('/api/purify', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
          document.getElementById('resultCard').style.display = 'block';
          document.getElementById('cleanPreview').src = data.cleaned_data_url;
          document.getElementById('downloadLink').href = data.cleaned_data_url;
          document.getElementById('downloadLink').download = 'cleaned_' + selectedFile.name;
          document.getElementById('psnrVal').innerText = data.metrics.psnr_db + ' dB';
          document.getElementById('ssimVal').innerText = data.metrics.ssim;
          document.getElementById('qualityVal').innerText = data.metrics.quality_preservation;
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
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/purify":
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

            if not file_bytes:
                self.send_json({"success": False, "error": "No image uploaded"}, 400)
                return

            # Process image in temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                input_file = os.path.join(tmpdir, "input.png")
                output_file = os.path.join(tmpdir, "output.png")

                with open(input_file, "wb") as f:
                    f.write(file_bytes)

                purifier = SynthIDPurifier(mode=mode, strength=strength)
                res = purifier.process_image(input_file, output_file, output_format="PNG")

                with open(output_file, "rb") as f:
                    cleaned_bytes = f.read()

                b64_img = base64.b64encode(cleaned_bytes).decode("utf-8")
                data_url = f"data:image/png;base64,{b64_img}"

                self.send_json({
                    "success": True,
                    "cleaned_data_url": data_url,
                    "metrics": res.get("metrics", {}),
                    "mode": mode,
                })
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def start_server(host: str = "127.0.0.1", port: int = 8080):
    server = HTTPServer((host, port), SynthIDWebHandler)
    print(f"🚀 SynthID Remover Web UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()


if __name__ == "__main__":
    start_server()
