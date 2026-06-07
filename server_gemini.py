"""
MediSnap — Gemini OCR Server (CLOUD version)
============================================
Same as your local server_gemini.py, but ready for cloud hosting (Render):
  - The API key is read ONLY from the GEMINI_API_KEY environment variable
    (so your key is NOT stored in the code / public repo).
  - Runs under gunicorn (Render start command), and also works locally.

Deploy steps are in SETUP_CLOUD_DEPLOY.md.
"""

import os
import re
import json
import datetime

from flask import Flask, request, jsonify
from google import genai
from google.genai import types

# API key comes from the hosting environment (set it in Render → Environment).
API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Free-tier model with a high daily limit. Alternatives: "gemini-2.5-flash", "gemini-3-flash".
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

client = genai.Client(api_key=API_KEY) if API_KEY else None
app = Flask(__name__)

PROMPT = """This is a photo of a blood pressure monitor (OMRON). Look only at the
LCD display numbers. Read the three values:
- SYS (systolic, the top/largest number)
- DIA (diastolic, the middle number)
- PULSE (heart rate, the bottom number)

Return ONLY a JSON object, nothing else:
{"systolic": 118, "diastolic": 78, "pulse": 70}

If a value is not clearly visible, use null. No explanation, no markdown."""


def extract_with_gemini(image_bytes):
    if client is None:
        print("[GEMINI] No API key set (GEMINI_API_KEY).")
        return {"systolic": None, "diastolic": None, "pulse": None, "spo2": None}
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), PROMPT],
        )
        text = re.sub(r"```json|```", "", (response.text or "").strip()).strip()
        data = json.loads(text)
        result = {"systolic": data.get("systolic"), "diastolic": data.get("diastolic"),
                  "pulse": data.get("pulse"), "spo2": None}
        print(f"[GEMINI] Parsed -> {result}")
        return result
    except json.JSONDecodeError:
        print(f"[GEMINI] Could not parse JSON: {getattr(response, 'text', '')}")
    except Exception as e:
        print(f"[GEMINI] Error: {e}")
    return {"systolic": None, "diastolic": None, "pulse": None, "spo2": None}


def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/")
def home():
    return "MediSnap OCR server is running. POST a photo to /ocr", 200


@app.route("/ocr", methods=["POST", "OPTIONS"])
def handle_ocr():
    if request.method == "OPTIONS":
        return _cors(app.make_response(("", 204)))
    image_bytes = request.get_data()
    if not image_bytes:
        return _cors(jsonify(error="No image data received")), 400
    print(f"\n[GEMINI] Received image ({len(image_bytes)} bytes), asking Gemini...")
    r = extract_with_gemini(image_bytes)
    return _cors(jsonify(systolic=r["systolic"], diastolic=r["diastolic"], pulse=r["pulse"], spo2=r["spo2"]))


if __name__ == "__main__":
    # Local run; on Render, gunicorn serves `app` instead.
    port = int(os.environ.get("PORT", 5000))
    print(f"=== MediSnap GEMINI Server (cloud build) on port {port} ===")
    app.run(host="0.0.0.0", port=port)
