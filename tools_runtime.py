# tools_runtime.py
import base64

# Minimal sign GIF map; expand as needed
SIGN_GIF_MAP = {
    "HELLO": "assets/hello.gif",
    "THANK-YOU": "assets/thankyou.gif",
}

def tool_sign_render(gloss: str) -> dict:
    url = SIGN_GIF_MAP.get(gloss.strip().upper(), "")
    return {"gloss": gloss, "gif_url": url}

def tool_safety_check(text: str) -> dict:
    """
    Ultra-light local placeholder for demo stability.
    Swap with Nemotron Safety Guard call if you want real moderation.
    """
    label = "pass"
    categories = []
    if any(word in text.lower() for word in ["hate", "violence"]):
        label = "warn"
        categories.append("toxicity")
    return {"label": label, "categories": categories}

def tool_riva_tts(text: str) -> dict:
    """
    Stub for Riva TTS; return empty audio in base64 for demo.
    Wire real gRPC later.
    """
    return {"wav_b64": ""}

def encode_image_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")