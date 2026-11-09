"""
Vercel Serverless Function Handler.
Vercel automatically routes HTTP requests to classes named 'handler'
inheriting from http.server.BaseHTTPRequestHandler.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthid_remover.web.server import SynthIDWebHandler

# Vercel entrypoint
handler = SynthIDWebHandler

