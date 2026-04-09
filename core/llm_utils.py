"""
ScholAR - Shared LLM utility.
Uses google-genai SDK with gemini-2.5-flash only.
"""

import os
import time
from google import genai
from google.genai import types
from rich.console import Console

console = Console()

MODEL = "gemini-2.5-flash"


def call_gemini(prompt: str, max_tokens: int = 4096, temperature: float = 0.3, max_retries: int = 2) -> str:
    """
    Call Gemini 2.5 Flash with retry on rate limits.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("No GEMINI_API_KEY set in environment")

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries):
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
            is_quota = "429" in err_msg or "quota" in err_msg.lower() or "rate" in err_msg.lower() or "RESOURCE_EXHAUSTED" in err_msg

            if is_quota and attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                console.print(f"[yellow]Rate limited, retrying in {wait_time}s...[/yellow]")
                time.sleep(wait_time)
            else:
                raise
