"""
voice2action.py
- Record mic audio -> Gemini transcribes -> Gemini returns Autopy + Keyboard code -> we validate + execute it
"""
from dotenv import load_dotenv
import os
import re
import time
import tempfile
import sounddevice as sd
import soundfile as sf

# Action libs we will expose to the generated code:
import autopy
import keyboard

# ---- Gemini (new Google Gen AI SDK) ----
# Audio understanding + Files API usage
from google import genai
from google.genai import types as genai_types

# -----------------------------
# 1) Audio capture
# -----------------------------
def record_audio_wav(path: str, seconds: float = 25.0, samplerate: int = 16000):
    """Record mono audio to WAV at 16kHz for a fixed duration."""
    print(f"🎙️  Recording for {seconds} sec in 3…")
    time.sleep(1)
    print("2…")
    time.sleep(1)
    print("1…")
    time.sleep(1)
    print("▶️  Speak now...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio, samplerate)
    print(f"💾 Saved audio to {path}")

# -----------------------------
# 2) Gemini helpers
# -----------------------------
def make_gemini_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")
    # Client for the Gemini Developer API
    client = genai.Client(api_key=api_key)
    return client

def transcribe_with_gemini(client: genai.Client, wav_path: str) -> str:
    """Upload audio and ask Gemini for a transcript in the original script."""
    # Upload audio
    myfile = client.files.upload(file=wav_path)

    prompt = (
        "You are a speech-to-text transcriber. "
        "Transcribe the audio exactly as spoken. "
        "If the spoken language is English, output in English. "
        "If the spoken language is not English, translate spoken language to English. "
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, myfile],
        config=genai_types.GenerateContentConfig(temperature=0.0),
    )

    transcript = (resp.text or "").strip()
    print(f"📝 Transcript: {transcript!r}")
    return transcript

# -----------------------------
# 3) Hard-coded UI elements (replace later with OmniParser)
#    Normalized bbox format: [x1, y1, x2, y2] in 0..1 coordinates.
# -----------------------------
UI_ELEMENTS = [
    {"id": 20, "type": "icon", "bbox": [0.2415, 0.3639, 0.4785, 0.4234], "interactivity": True, "content": "Type your name"},
    {"id": 18, "type": "icon", "bbox": [0.5120, 0.3666, 0.5751, 0.4196], "interactivity": True, "content": "Email"},
    {"id": 19, "type": "icon", "bbox": [0.5808, 0.3667, 0.6476, 0.4194], "interactivity": True, "content": "Phone"},
    {"id": 17, "type": "icon", "bbox": [0.2421, 0.5527, 0.4795, 0.6748], "interactivity": True, "content": "Type your message here.."},
    {"id": 16, "type": "icon", "bbox": [0.5104, 0.5523, 0.7490, 0.6773], "interactivity": True, "content": "Share your feedback."},
    {"id": 21, "type": "icon", "bbox": [0.2447, 0.7994, 0.3012, 0.8506], "interactivity": True, "content": "Submit"},
]

# -----------------------------
# 4) Ask Gemini to generate Autopy + Keyboard code
# -----------------------------
GEN_CODE_SYSTEM_INSTRUCTION = (
    "You are an automation code generator. "
    "Return ONLY a Python fenced code block labeled 'python' that can be executed as-is. "
    "Requirements:\n"
    "- Use ONLY these libraries: autopy (for mouse) and keyboard (for typing/keys) and time for delays.\n"
    "- DO NOT use autopy.key.* APIs. Use keyboard.write(...) / keyboard.send('enter') etc for typing.\n"
    "- Expect a dict named UI_ELEMENTS with normalized bboxes [x1,y1,x2,y2] in [0,1]; "
    "  to click, convert bbox center to pixels using autopy.screen.size().\n"
    "- After moving/clicking, wait ~0.3-0.8s before typing/next action for UI stability.\n"
    "- Be robust: double-click the textbox before typing to ensure focus; "
    "  after opening dropdown, wait 0.8s, then click the Pakistan option; then click Submit.\n"
    "- Do not import anything else (no os, sys, subprocess, eval, open, __import__).\n"
    "- Whenever user commands to switch text input box or area add an extra space."
    "- To click: first use autopy.mouse.move(x, y), then call autopy.mouse.click().\n"
    "- Do NOT pass x,y to autopy.mouse.click(); it only takes an optional button argument.\n"
    "- Print 'Automation done' at the end."
)

def build_code_prompt(transcript: str) -> str:
    return (
        f"{GEN_CODE_SYSTEM_INSTRUCTION}\n\n"
        f"User instruction (transcribed): {transcript}\n\n"
        "UI elements (normalized bboxes JSON):\n"
        f"{UI_ELEMENTS}\n\n"
        "Write the code now."
    )

def request_code_from_gemini(client: genai.Client, prompt: str) -> str:
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0.2),
    )
    raw = resp.text or ""
    # Extract fenced block correctly
    m = re.search(r"```python\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1).strip()
    else:
        code = raw.strip()
    if not code:
        raise RuntimeError("Model did not return Python code.")
    return code

# -----------------------------
# 5) Minimal sandboxing & execution
# -----------------------------
ALLOWED_IMPORTS = {"autopy", "keyboard", "time"}

def is_code_safe(code):
    BLOCKED_PATTERNS = [
        "import os",
        "import subprocess",
        "open(",
        "eval(",
        "exec(",
    ]
    for pat in BLOCKED_PATTERNS:
        if pat in code:
            return False, f"Blocked pattern found: {pat}"
    return True, "Code is safe"

HELPERS = """
# --- helpers injected by the runner ---
SCREEN_WIDTH, SCREEN_HEIGHT = autopy.screen.size()
def bbox_to_center(b):
    x1, y1, x2, y2 = b
    cx = int(((x1 + x2) / 2) * SCREEN_WIDTH)
    cy = int(((y1 + y2) / 2) * SCREEN_HEIGHT)
    return cx, cy
"""

def run_generated_code(code: str):
    safe, msg = is_code_safe(code)
    if not safe:
        raise RuntimeError(f"Refusing to execute untrusted code: {msg}")
    # Build the execution environment
    g = {
        "autopy": autopy,
        "keyboard": keyboard,
        "time": time,
        "UI_ELEMENTS": UI_ELEMENTS,
    }
    # Prepend helpers so generated code can call bbox_to_center and use screen size.
    full_code = HELPERS + "\n\n" + code
    print("▶️ Executing generated code...")
    exec(full_code, g, None)

# -----------------------------
# 6) Main
# -----------------------------
def main():
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "user_command.wav")
        record_audio_wav(wav_path, seconds=25)
        client = make_gemini_client()
        transcript = transcribe_with_gemini(client, wav_path)

        prompt = build_code_prompt(transcript)
        print("🧠 Requesting automation code from Gemini…")
        code = request_code_from_gemini(client, prompt)

        print("----- GENERATED CODE START -----")
        print(code)
        print("----- GENERATED CODE END -----")

        run_generated_code(code)
        print("✅ Done")

if __name__ == "__main__":
    main()
