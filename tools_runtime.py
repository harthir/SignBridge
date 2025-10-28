# tools_runtime.py
import base64

# Minimal sign GIF map; expand if you add more assets
SIGN_GIF_MAP = {
    "HELLO": "assets/hello.gif",
    "THANK-YOU": "assets/thankyou.gif",
}

def tool_sign_render(gloss: str) -> dict:
    url = SIGN_GIF_MAP.get(gloss.strip().upper(), "")
    return {"gloss": gloss, "gif_url": url}

def tool_safety_check(text: str) -> dict:
    """
    Lightweight local moderation placeholder.
    Swap with Safety-Guard via NIM if desired.
    """
    label = "pass"
    cats = []
    bad = ["hate", "kill", "violence", "slur"]
    if any(w in text.lower() for w in bad):
        label = "warn"
        cats.append("toxicity")
    return {"label": label, "categories": cats}

def tool_riva_tts(text: str) -> dict:
    """
    Stub for TTS; return empty audio (keeps demo flowing on py3.13).
    """
    return {"wav_b64": ""}

def encode_image_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")