"""Provider configuration, chat-completion calls, and optional web snippets."""
import html
import os
import random
import re
import time
import urllib.parse
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


def create_client(provider: str, base_url: str | None = None):
    """Create an OpenAI-compatible client without embedding API keys in code."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the API client with `pip install openai`.") from exc
    provider = provider.strip().lower()
    if provider == "ollama":
        url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    elif provider == "openai-compatible":
        url = base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in your environment for the "
                "openai-compatible provider, or use --provider ollama."
            )
    else:
        raise ValueError(f"Unsupported provider: {provider!r}")
    return OpenAI(api_key=api_key, base_url=url)


def _llm_call(client, model: str, prompt: str, system: str = "", max_tokens: int = 512,
              provider: str = "openai-compatible") -> str:
    """Call the configured chat-completions endpoint, retrying transient failures."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(3):
        try:
            kwargs = {
                "model": model,
                "temperature": 0.0,
                "messages": messages,
                "stream": False,
            }
            if provider == "ollama":
                kwargs.update({
                    "max_tokens": max_tokens,
                    "extra_body": {"options": {"num_predict": max_tokens}},
                })
            else:
                kwargs["max_completion_tokens"] = max_tokens
            response = client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            wait = min(30, (2 ** attempt) + random.random())
            print(f"  ⚠️ LLM call failed ({type(exc).__name__}: {exc}); retry "
                  f"{attempt + 1}/3 in {wait:.1f}s", flush=True)
            time.sleep(wait)
    return ""


def _web_search_snippets(query: str, max_results: int = 5, timeout: int = 10) -> str:
    """Best-effort web snippets from DuckDuckGo HTML results."""
    try:
        q = urllib.parse.quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")

        links = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not snippets:
            snippets = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</div>',
                body,
                flags=re.IGNORECASE | re.DOTALL,
            )

        lines = []
        for idx, (href, title_html) in enumerate(links[: max_results * 2]):
            src = href
            parsed = urllib.parse.urlparse(href)
            if "duckduckgo.com" in (parsed.netloc or ""):
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs and qs["uddg"]:
                    src = urllib.parse.unquote(qs["uddg"][0])

            title = re.sub(r"<[^>]+>", "", html.unescape(title_html)).strip()
            snippet = ""
            if idx < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", html.unescape(snippets[idx])).strip()
            text = re.sub(r"\s+", " ", f"{title}. {snippet}").strip(" .")
            if text:
                lines.append(f"- {text} (source: {src})")

        return "\n".join(lines[:max_results]).strip()
    except Exception as e:
        print(f"  ⚠️  Web search failed: {type(e).__name__}: {e}", flush=True)
        return ""
