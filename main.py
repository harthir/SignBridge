# main.py
import os, json, re, base64, requests
from typing import TypedDict, List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from tools_runtime import (
    tool_sign_render, tool_safety_check, tool_riva_tts, encode_image_b64
)

load_dotenv()

# ---------- ENV ----------
NIM_BASE_URL   = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_VL_MODEL   = os.getenv("NIM_VL_MODEL", "nvidia/nemotron-nano-12b-v2-vl")
NIM_TEXT_MODEL = os.getenv("NIM_TEXT_MODEL", "nvidia/nemotron-nano-12b-v2")

# You can use a single key; shown as two for clarity.
NIM_API_KEY_VL   = os.getenv("NIM_API_KEY_VL") or os.getenv("NIM_API_KEY")
NIM_API_KEY_TEXT = os.getenv("NIM_API_KEY_TEXT") or os.getenv("NIM_API_KEY")

assert NIM_API_KEY_VL,   "Missing NIM_API_KEY_VL (or NIM_API_KEY)"
assert NIM_API_KEY_TEXT, "Missing NIM_API_KEY_TEXT (or NIM_API_KEY)"

# ---------- APP ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"ok": True}

# ---------- NIM helpers ----------
def nim_chat(model: str, api_key: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    Raw call to /chat/completions (OpenAI-compatible). Returns raw JSON response.
    """
    url = f"{NIM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 256),
        "temperature": kwargs.get("temperature", 0.2),
        "top_p": kwargs.get("top_p", 1),
        "stream": False,
        # pass tools if provided
    }
    if "tools" in kwargs:
        payload["tools"] = kwargs["tools"]

    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()

def classify_frame_bytes(img_bytes: bytes, query: str = "Classify the ASL sign in this frame.") -> Dict[str, Any]:
    """
    Use Nemotron-Nano-12B-v2-VL to return {"gloss": "...", "confidence": 0..1}
    """
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    content = [
        {"type": "text", "text": (
            "You are an ASL vision classifier. "
            "Return ONLY compact JSON with keys gloss (string) and confidence (0..1). "
            'Format: {\"gloss\":\"...\",\"confidence\": 0.xx}. No extra text.\n\n' + query
        )},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    ]
    messages = [
        {"role": "system", "content": "/think"},
        {"role": "user", "content": content}
    ]
    raw = nim_chat(NIM_VL_MODEL, NIM_API_KEY_VL, messages, max_tokens=256, temperature=0.2)
    text = raw["choices"][0]["message"]["content"]
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        obj = {"gloss": "UNKNOWN", "confidence": 0.0, "raw": text}
    g = obj.get("gloss", "").strip().upper().replace(" ", "-")
    c = float(obj.get("confidence", 0.0) or 0.0)
    return {"gloss": g, "confidence": max(0.0, min(c, 1.0))}

def translate_gloss_to_english(gloss: str) -> str:
    messages = [
        {"role":"system","content":"Convert ASL gloss into brief, natural English. Output English only."},
        {"role":"user","content": f'ASL gloss: "{gloss}"'}
    ]
    raw = nim_chat(NIM_TEXT_MODEL, NIM_API_KEY_TEXT, messages, max_tokens=64, temperature=0.2)
    return raw["choices"][0]["message"]["content"].strip()

# ---------- Function-calling (planner) ----------
# Tool schemas (OpenAI-style). The LLM will decide which to call; we execute and feed results back.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "classify_sign",
            "description": "Classify an ASL frame; returns gloss and confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string", "description": "JPEG/PNG base64 image without the data: prefix"}
                },
                "required": ["image_b64"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sign_render",
            "description": "Get a GIF/clip URL for a given gloss.",
            "parameters": {
                "type": "object",
                "properties": {"gloss": {"type":"string"}},
                "required": ["gloss"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "safety_check",
            "description": "Moderate text; returns pass/warn/block and categories.",
            "parameters": {
                "type": "object",
                "properties": {"text":{"type":"string"}},
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "riva_tts",
            "description": "Text-to-speech; returns Base64 WAV.",
            "parameters": {
                "type": "object",
                "properties": {"text":{"type":"string"}},
                "required": ["text"]
            }
        }
    }
]

PLANNER_SYSTEM = (
    "You are the Coordinator/Translator Agent for ASL ↔ English.\n"
    "Follow a Reason→Act→Observe loop and use tools when needed.\n"
    "Policy:\n"
    "- If sign classification confidence < 0.6, ask for another frame (do not fabricate).\n"
    "- For Speech→Sign, produce brief, clause-level English suitable for signing.\n"
    "- Always run safety_check before finalizing text or speech.\n"
    "Output JSON envelope in the final assistant message: "
    '{"plan":"..","calls":[...],"observations":[...],"final_text":"...","final_audio_b64":""}\n'
)

def run_planner_with_tools(user_goal: str, image_b64: Optional[str] = None) -> Dict[str, Any]:
    """
    Single conversation turn with tool-calling. We allow up to 3 tool iterations.
    """
    messages: List[Dict[str, Any]] = [{"role": "system", "content": PLANNER_SYSTEM}]
    # Seed message describes available context & input
    seed = f"Goal: {user_goal}"
    if image_b64:
        seed += "\nWe also have an ASL image frame available as base64."
    messages.append({"role": "user", "content": seed})

    loop_count = 0
    calls_log: List[Dict[str, Any]] = []
    observations: List[str] = []
    final_payload: Dict[str, Any] = {}

    while loop_count < 3:
        resp = nim_chat(
            NIM_TEXT_MODEL,
            NIM_API_KEY_TEXT,
            messages,
            tools=TOOLS_SCHEMA,
            max_tokens=512,
            temperature=0.2
        )
        msg = resp["choices"][0]["message"]

        # If the model directly produced the final JSON, return it:
        if "tool_calls" not in msg:
            # Try to parse an envelope if present
            content = msg.get("content", "")
            try:
                cleaned = re.sub(r"```json|```", "", content).strip()
                final_payload = json.loads(cleaned)
            except Exception:
                final_payload = {"plan": "no-tools", "calls": calls_log, "observations": observations, "final_text": content}
            break

        # Execute tool calls:
        for tc in msg["tool_calls"]:
            fn = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            result = {}

            if fn == "classify_sign":
                ib64 = args.get("image_b64")
                if not ib64 and image_b64:
                    ib64 = image_b64  # allow planner to omit arg; we supply captured frame
                if not ib64:
                    result = {"error":"missing image_b64"}
                else:
                    result = classify_frame_bytes(base64.b64decode(ib64))

            elif fn == "sign_render":
                result = tool_sign_render(args.get("gloss",""))

            elif fn == "safety_check":
                result = tool_safety_check(args.get("text",""))

            elif fn == "riva_tts":
                result = tool_riva_tts(args.get("text",""))

            else:
                result = {"error": f"unknown tool {fn}"}

            # Log calls
            calls_log.append({"name": fn, "args": args, "result": result})
            observations.append(f"{fn}→{json.dumps(result)[:180]}")

            # Feed tool result back
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", fn),
                "name": fn,
                "content": json.dumps(result)
            })

        loop_count += 1

    if not final_payload:
        # If we never got a final envelope, synthesize one
        final_payload = {
            "plan": "tool-loop-ended",
            "calls": calls_log,
            "observations": observations,
            "final_text": calls_log[-1]["result"].get("gloss","") if calls_log else ""
        }
    return final_payload

# ---------- FastAPI endpoints ----------

class GlossIn(BaseModel):
    gloss: str

class STSIn(BaseModel):
    text: str

@app.post("/asl/frame")
async def asl_frame(file: UploadFile = File(...)):
    img_bytes = await file.read()
    return classify_frame_bytes(img_bytes)

@app.post("/translate/gloss")
async def translate_gloss(body: GlossIn):
    try:
        eng = translate_gloss_to_english(body.gloss)
    except Exception:
        eng = "Hello!"
    return {"english": eng}

@app.post("/run/sign_to_speech")
async def run_sign_to_speech(file: UploadFile = File(...)):
    """
    Full agentic path using function-calling:
    - Planner decides to call classify_sign (we pass image_b64)
    - Planner calls safety_check, riva_tts, etc.
    """
    img = await file.read()
    b64 = encode_image_b64(img)
    envelope = run_planner_with_tools(
        user_goal="Translate the ASL sign into natural English and speak it.",
        image_b64=b64
    )
    # if planner skipped audio, add a no-audio field
    envelope.setdefault("final_audio_b64", "")
    return envelope

@app.post("/run/speech_to_sign")
async def run_speech_to_sign(body: STSIn):
    """
    Planner can skip classify_sign and go straight to safety_check + sign_render.
    """
    # We let the planner decide, but give it a nudge by including the goal
    envelope = run_planner_with_tools(
        user_goal=f'Convert this phrase into signing-friendly English and map to a sign if possible: "{body.text}"'
    )
    envelope.setdefault("sign_animation_url", "")
    # If a sign_render call happened, surface its gif_url:
    for c in envelope.get("calls", []):
        if c.get("name") == "sign_render" and isinstance(c.get("result"), dict):
            envelope["sign_animation_url"] = c["result"].get("gif_url", "")
    return envelope