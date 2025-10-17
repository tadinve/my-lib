"""Favorite utility functions.

Keep these small and well-tested; consumers can pip install from git.
"""

import os
import sys
def wrap_text(text,numchars=100):
    lines = text.split('\n')
    for line in lines:
        while len(line) > numchars:
            space_index = line[:numchars].rfind(' ')
            if space_index == -1:  # no space found
                print(line[:numchars])
                line = line[numchars:]
            else:
                print(line[:space_index])
                line = line[space_index+1:]
        print(line)
    return



def load_colab_keys(keys = ["GOOGLE_API_KEY", "OPENAI_API_KEY"]):
    """Safely load API keys from Colab userdata if running in Colab."""
    for key in keys:
        try:
            from google.colab import userdata  # type: ignore
            try:
                g_key = userdata.get(key)
                if g_key:
                    os.environ[key] = g_key
                    print(f"🔑 Loaded {key} from userdata")
            except userdata.SecretNotFoundError:
                    print(f"🔑 {key} not defined in userdata")

        except ImportError:
            pass  # Not in Colab
    return

import os
from typing import Callable

def make_llm() -> Callable[[str], str]:
    """Return a callable(prompt: str) -> str using Gemini, OpenAI, or stub.
    In Colab, automatically fetch keys from secrets if available.
    """
    load_colab_keys()


    if os.getenv("GOOGLE_API_KEY"):
        try:
            from google import genai
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

            def gemini_call(prompt: str) -> str:
                # ✅ pass a plain string (or list of strings), not role/parts dicts
                resp = client.models.generate_content(
                    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), # Changed model name to gemini-1.0-pro
                    contents=prompt,
                )
                # Be defensive in extraction across SDK versions
                try:
                    # new SDK shape
                    return resp.candidates[0].content.parts[0].text.strip()
                except Exception:
                    # fallback: try generic attributes
                    if hasattr(resp, "text") and resp.text:
                        return resp.text.strip()
                    return str(resp)

            print("🔹 Using Gemini (1.0-Pro or configured model)") # Updated print message
            return gemini_call
        except Exception as e:
            print("⚠️ Gemini initialization failed:", e)

    # --- OpenAI next ---
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            def openai_call(prompt: str) -> str:
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                return resp.choices[0].message.content.strip()
            print("🔹 Using OpenAI (gpt-4o-mini or configured model)")
            return openai_call
        except Exception as e:
            print("⚠️ OpenAI initialization failed:", e)

    # --- Stub fallback ---
    def stub_call(prompt: str) -> str:
        s = prompt.replace("\n", " ")
        return "STUB-LLM → " + (s[:140] + ("..." if len(s) > 140 else ""))
    print("🪄 Using Stub LLM (offline mode)")
    return stub_call