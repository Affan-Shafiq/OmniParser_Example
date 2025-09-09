"""
voice2action.py
- Record mic audio -> Gemini transcribes + generates Autopy + Keyboard code -> we validate + execute it
"""
from dotenv import load_dotenv
import os
import re
import time
import tempfile
import sounddevice as sd
import soundfile as sf

# Action libs
import autopy
import keyboard

# ---- Gemini (new Google Gen AI SDK) ----
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
    return genai.Client(api_key=api_key)

# -----------------------------
# 3) Hard-coded UI elements
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
# 4) One-shot Gemini request
# -----------------------------
SYSTEM_PROMPT = (
    "You are an automation assistant.\n"
    "Task:\n"
    "1. First, transcribe the uploaded audio exactly as spoken.\n"
    "   - If English, keep English.\n"
    "   - If non-English, translate into English.\n"
    "   - Return the transcription clearly marked.\n\n"
    "2. Then, using that transcription and the provided UI_ELEMENTS JSON, "
    "   generate Python automation code.\n\n"
    "Automation code rules:\n"
    "- Use only: autopy (mouse), keyboard (typing), and time (delays).\n"
    "- No autopy.key APIs. For typing use keyboard.write() / keyboard.send().\n"
    "- Convert bbox center to pixels with autopy.screen.size().\n"
    "- Double-click textboxes before typing.\n"
    "- Wait 0.3–0.8s after actions.\n"
    "- Print 'Automation done' at the end.\n"
    "- Return ONLY one fenced Python code block labeled ```python ... ```.\n"
    "Important safeguard:\n"
     "When generating code with autopy:\n"
        "- Always move the mouse first with autopy.mouse.move(x, y).\n"
        "- Then call autopy.mouse.click(button=autopy.mouse.Button.LEFT, count=1 or 2).\n"
        "- Do NOT pass x, y to autopy.mouse.click(), because it only accepts (button, count).\n"
        "- Example:\n"
            "autopy.mouse.move(x, y)\n"
            "autopy.mouse.click(autopy.mouse.Button.LEFT, 1)\n"
)

def transcribe_and_generate(client: genai.Client, wav_path: str) -> tuple[str, str]:
    """One Gemini call: transcription + automation code."""
    myfile = client.files.upload(file=wav_path)

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[SYSTEM_PROMPT, f"UI elements: {UI_ELEMENTS}", myfile],
        config=genai_types.GenerateContentConfig(temperature=0.2),
    )

    raw = resp.text or ""
    # Extract transcript (before code block)
    transcript_part = re.split(r"```python", raw, 1)[0].strip()
    transcript = transcript_part.replace("Transcription:", "").strip()

    # Extract code block
    m = re.search(r"```python\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        raise RuntimeError("Model did not return Python code.")
    code = m.group(1).strip()

    print(f"📝 Transcript: {transcript!r}")
    return transcript, code

# -----------------------------
# 5) Minimal sandboxing & execution
# -----------------------------
def is_code_safe(code):
    BLOCKED = ["import os", "import subprocess", "open(", "eval(", "exec("]
    for pat in BLOCKED:
        if pat in code:
            return False, f"Blocked pattern: {pat}"
    return True, "Code safe"

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
        raise RuntimeError(f"Unsafe code: {msg}")
    g = {"autopy": autopy, "keyboard": keyboard, "time": time, "UI_ELEMENTS": UI_ELEMENTS}
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

        transcript, code = transcribe_and_generate(client, wav_path)

        print("----- GENERATED CODE START -----")
        print(code)
        print("----- GENERATED CODE END -----")

        run_generated_code(code)
        print("✅ Done")

if __name__ == "__main__":
    main()
