"""
Entrypoint for Cloud Deployment (Hugging Face Spaces, Render, Koyeb, Docker).
"""

import os
from synthid_remover.web.server import start_server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # 7860 is HF Spaces default port
    start_server(host="0.0.0.0", port=port)

