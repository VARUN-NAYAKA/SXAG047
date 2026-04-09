"""
ScholAR - Shared LLM utility.
Uses google.genai SDK (new) with gemini-2.0-flash.
Includes rate limiting to avoid quota exhaustion.
"""

import os
import time
import threading
from google import genai
from google.genai import types
from rich.console import Console

console = Console()

MODEL = "gemini-flash-latest"

# ── Rate limiter: max 8 requests per minute (free tier safety margin) ──
_lock = threading.Lock()
_call_times: list[float] = []
_MAX_RPM = 8


def _rate_limit():
    """Ensure we don't exceed RPM quota."""
    with _lock:
        now = time.time()
        # Remove calls older than 60s
        _call_times[:] = [t for t in _call_times if now - t < 60]
        if len(_call_times) >= _MAX_RPM:
            wait = 60 - (now - _call_times[0]) + 1
            console.print(f"[yellow]Rate limiter: waiting {wait:.0f}s to stay under quota...[/yellow]")
            time.sleep(wait)
        _call_times.append(time.time())


def call_gemini(prompt: str, max_tokens: int = 4096, temperature: float = 0.3, max_retries: int = 5) -> str:
    """
    Call Gemini 2.0 Flash with built-in rate limiting and retry.
    Reads API key from environment variable or uses default.
    """
    # Try environment first, then fall back to default
    api_key = os.environ.get("GEMINI_API_KEY") or "AIzaSyAl6eyojROioBI3A8Jxok_sXFC_896-Gxo"

    if not api_key:
        raise ValueError("No GEMINI_API_KEY set in environment or config")

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries):
        _rate_limit()
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text.strip()

        except Exception as e:
            err_msg = str(e)

            # Check for quota exhaustion
            if "429" in err_msg and "limit: 0" in err_msg:
                raise RuntimeError(
                    "🔴 FREE TIER QUOTA EXHAUSTED\n\n"
                    "The default API key has hit its daily free tier limit (0 remaining).\n\n"
                    "SOLUTION: Paste your own Gemini API key in the Streamlit sidebar:\n"
                    "1. Get a paid Gemini API key from https://ai.google.dev/\n"
                    "2. Enter it in the 'Custom Gemini API Key' field in the left sidebar\n"
                    "3. Retry the search\n\n"
                    "The free tier quota resets daily at UTC midnight."
                ) from e

            is_retryable = (
                "429" in err_msg or
                "quota" in err_msg.lower() or
                "RESOURCE_EXHAUSTED" in err_msg or
                "503" in err_msg or
                "UNAVAILABLE" in err_msg
            )

            if is_retryable and attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                console.print(f"[yellow]Service temporarily unavailable (attempt {attempt+1}), waiting {wait_time}s...[/yellow]")
                time.sleep(wait_time)
            else:
                raise
