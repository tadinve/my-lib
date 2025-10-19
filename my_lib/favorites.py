"""
Favorite utility functions.

Keep these small and well-tested; consumers can pip install from git.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Optional

# -----------------------
# Text helpers
# -----------------------
def wrap_text(text: str, numchars: int = 100, print_: bool = False) -> str:
    """Soft-wrap text at numchars. Returns the wrapped string (optionally prints)."""
    if numchars <= 0:
        return text
    out_lines = []
    for raw in text.splitlines():
        line = raw
        while len(line) > numchars:
            cut = line[:numchars].rfind(" ")
            if cut == -1:
                out_lines.append(line[:numchars])
                line = line[numchars:]
            else:
                out_lines.append(line[:cut])
                line = line[cut + 1 :]
        out_lines.append(line)
    wrapped = "\n".join(out_lines)
    if print_:
        print(wrapped)
    return wrapped


# -----------------------
# Secrets loader (Colab)
# -----------------------
def load_colab_keys(keys: Iterable[str] = ("GOOGLE_API_KEY", "OPENAI_API_KEY")) -> None:
    """
    Safely load API keys from Colab userdata if running in Colab.
    Falls back silently when not in Colab or key absent.
    """
    try:
        from google.colab import userdata  # type: ignore

        for key in keys:
            try:
                val = userdata.get(key)
                if val:
                    os.environ[key] = val
                    print(f"🔑 Loaded {key} from Colab userdata")
            except Exception:
                # SecretNotFoundError or any retrieval error
                pass
    except Exception:
        # Not in Colab
        pass


# -----------------------
# LLM selector
# -----------------------
def make_llm(
    keys: Iterable[str] = ("GOOGLE_API_KEY", "OPENAI_API_KEY"),
    *,
    provider: Optional[str] = None,   # "gemini" | "openai" | "stub" | None(auto)
    dry_run: bool = False             # force stub even if keys exist
) -> Callable[[str], str]:
    """
    Return a callable(prompt: str) -> str using Gemini, OpenAI, or stub.

    - Auto-loads Colab secrets if present.
    - If provider is set, tries that first; else priority: Gemini -> OpenAI -> Stub.
    - dry_run=True forces stub (useful in class for deterministic demos).
    """
    load_colab_keys(keys)

    if dry_run:
        print("🧪 DRY_RUN enabled — forcing Stub LLM")
        return _stub_llm()

    # If a provider is requested, honor it
    if provider == "gemini":
        llm = _try_gemini()
        if llm:
            return llm
        print("⚠️ Requested provider 'gemini' unavailable; falling back.")
    elif provider == "openai":
        llm = _try_openai()
        if llm:
            return llm
        print("⚠️ Requested provider 'openai' unavailable; falling back.")
    elif provider == "stub":
        return _stub_llm()

    # Auto: Gemini -> OpenAI -> Stub
    llm = _try_gemini()
    if llm:
        return llm
    llm = _try_openai()
    if llm:
        return llm
    return _stub_llm()


# -----------------------
# Providers
# -----------------------
def _try_gemini() -> Optional[Callable[[str], str]]:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)

        # Pick a supported model. Prefer env override, else discover.
        def _pick_model() -> str:
            env = os.getenv("GEMINI_MODEL")
            if env:
                return env
            names = [m.name for m in client.models.list()]  # e.g., "models/gemini-1.5-flash-latest"
            # Prefer 2.x or 1.5 flash/pro latest
            for fam in ("gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash", "gemini-1.5-pro"):
                cands = [n for n in names if fam in n]
                for suf in ("-latest", "-001"):
                    for n in cands:
                        if suf in n:
                            return n
                if cands:
                    return cands[0]
            # last resort
            return names[0] if names else "models/gemini-1.5-flash-latest"

        model_name = _pick_model()
        print(f"🔹 Using Gemini: {model_name}")

        def gemini_call(prompt: str) -> str:
            # New SDK: pass a plain string; not role/parts
            resp = client.models.generate_content(model=model_name, contents=prompt)
            # Robust extraction across SDK variants
            try:
                return resp.candidates[0].content.parts[0].text.strip()
            except Exception:
                txt = getattr(resp, "text", None)
                return txt.strip() if isinstance(txt, str) and txt else str(resp)

        return gemini_call
    except Exception as e:
        print("⚠️ Gemini initialization failed:", e)
        return None


def _try_openai() -> Optional[Callable[[str], str]]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI()

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"🔹 Using OpenAI: {model}")

        def openai_call(prompt: str) -> str:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return r.choices[0].message.content.strip()

        return openai_call
    except Exception as e:
        print("⚠️ OpenAI initialization failed:", e)
        return None


def _stub_llm() -> Callable[[str], str]:
    print("🪄 Using Stub LLM (offline mode)")
    def stub_call(prompt: str) -> str:
        s = prompt.replace("\n", " ")
        return "STUB-LLM → " + (s[:140] + ("..." if len(s) > 140 else ""))
    return stub_call
