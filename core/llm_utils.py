"""
ScholAR - Shared LLM utility with retry and model fallback.
Uses the new google-genai SDK for proper model support.
"""

import os
import time
from google import genai
from google.genai import types
from rich.console import Console

console = Console()

# Models to try in order (fallback chain)
FALLBACK_MODELS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
]


def call_gemini(prompt: str, max_tokens: int = 4096, temperature: float = 0.3, max_retries: int = 2) -> str:
    """
    Call Gemini LLM with automatic retry and model fallback.
    Tries multiple models if quota is exhausted on one.
    Returns the response text.
    Raises RuntimeError if all models fail.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("No GEMINI_API_KEY set in environment")

    client = genai.Client(api_key=api_key)
    last_error = None

    for model_name in FALLBACK_MODELS:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text.strip()

            except Exception as e:
                last_error = e
                err_msg = str(e)
                is_quota = "429" in err_msg or "quota" in err_msg.lower() or "rate" in err_msg.lower() or "RESOURCE_EXHAUSTED" in err_msg

                if is_quota and attempt < max_retries - 1:
                    wait_time = 15 * (attempt + 1)
                    console.print(f"[yellow]Rate limited on {model_name}, retrying in {wait_time}s...[/yellow]")
                    time.sleep(wait_time)
                else:
                    console.print(f"[red]{model_name} failed: {str(e)[:100]}[/red]")
                    break  # Try next model

    raise RuntimeError(f"All Gemini models exhausted. Last error: {last_error}")
