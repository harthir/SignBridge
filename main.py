import os, json, re
from typing import Dict, Any, List
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---- ENV ----
NIM_BASE_URL   = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_TEXT_MODEL = os.getenv("NIM_TEXT_MODEL", "nvidia/nemotron-nano-12b-v2")
NIM_API_KEY_TEXT = os.getenv("NIM_API_KEY_TEXT") or os.getenv("NIM_API_KEY")
SIGNBRIDGE_FAKE = os.getenv("SIGNBRIDGE_FAKE", "0") == "1"

# ---- APP ----
app = FastAPI(title="SignBridge (MediaPipe → Nemotron)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "fake": SIGNBRIDGE_FAKE,
        "text_model": NIM_TEXT_MODEL,
        "has_text_key": bool(NIM_API_KEY_TEXT),
    }

# ---- NIM helper ----
def nim_chat(model: str, api_key: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    if SIGNBRIDGE_FAKE or not api_key:
        return {"choices":[{"message":{"content":"Hello! (demo mode)"}}]}

    url = f"{NIM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 64),
        "temperature": kwargs.get("temperature", 0.2),
        "top_p": kwargs.get("top_p", 1),
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # show the server message for easy debugging
        raise RuntimeError(f"NIM HTTP {r.status_code}: {r.text[:400]}") from e
    return r.json()

class GlossIn(BaseModel):
    gloss: str

@app.post("/translate/gloss")
def translate_gloss(body: GlossIn):
    """
    Convert an ASL-style gloss (e.g., 'THUMBS_UP', 'ONE', 'OPEN_PALM')
    into short, natural English using Nemotron.
    """
    gloss = (body.gloss or "").strip().upper().replace(" ", "_")

    # Simple fallback dictionary (if FAKE mode or model unreachable)
    fallback = {
        "ONE": "One.",
        "TWO": "Two.",
        "THREE": "Three.",
        "FOUR": "Four.",
        "FIVE": "Five.",
        "THUMBS_UP": "Yes / Good.",
        "FIST": "No / Stop.",
        "OPEN_PALM": "Hello!"
    }

    try:
        messages = [
            {"role":"system","content":"You turn ASL gloss tokens into short, natural English. Be concise."},
            {"role":"user","content": f'ASL gloss: "{gloss}"'}
        ]
        raw = nim_chat(NIM_TEXT_MODEL, NIM_API_KEY_TEXT, messages, max_tokens=40, temperature=0.1)
        text = raw["choices"][0]["message"]["content"].strip()
        return {"english": text or fallback.get(gloss, gloss.title())}
    except Exception as e:
        return {"english": fallback.get(gloss, "Hello!"), "note": str(e)}