"""Ollama client and tolerant JSON parsing.

All LLM traffic goes through this one interface so alternative backends (and
canned responses in offline checks) can be injected without touching the
pipeline. :class:`LLMClient` is the contract; :class:`OllamaClient` is the
real implementation talking to Ollama's local REST API.

Behaviours mandated by AGENTS.md ("LLM usage"):

* ``num_ctx`` is always set explicitly — Ollama's implicit default is tiny and
  silently truncates.
* ``keep_alive`` keeps the model resident between chunks.
* Thinking is disabled where the server supports it, and ``<think>...</think>``
  blocks are stripped before parsing regardless.
* JSON parsing tolerates leading/trailing junk and markdown fences.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from dramasub.core.errors import LLMError

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")


class LLMClient:
    """Interface for text/JSON generation. Subclass to inject responses."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        format_json: bool = False,
        num_ctx: int | None = None,
    ) -> str:
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        num_ctx: int | None = None,
    ) -> Any:
        """Generate and parse a JSON value, tolerating think tags and fences."""
        raw = self.generate(
            prompt,
            system=system,
            temperature=temperature,
            format_json=True,
            num_ctx=num_ctx,
        )
        return parse_json(raw)


@dataclass
class OllamaClient(LLMClient):
    """Talks to Ollama's ``/api/generate`` endpoint via ``requests``."""

    host: str = "http://localhost:11434"
    model: str = "qwen3.6:latest"
    num_ctx: int = 16384
    keep_alive: str = "30m"
    think: bool = False
    timeout: float = 300.0
    max_retries: int = 3
    backoff: float = 1.5
    session: requests.Session = field(default_factory=requests.Session, repr=False)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        format_json: bool = False,
        num_ctx: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": self.think,
            "options": {
                "num_ctx": num_ctx or self.num_ctx,
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        data = self._post_with_retries(payload)
        response = data.get("response")
        if not isinstance(response, str) or not response.strip():
            raise LLMError(f"empty response from model {self.model!r}")
        return response

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.host.rstrip('/')}/api/generate"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Ollama request failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                self._sleep(attempt)
                continue

            if resp.status_code == 400 and "think" in payload and _mentions_think(resp):
                # Model/server doesn't accept the `think` flag — drop and retry.
                logger.info("model %r rejected think flag; retrying without it", self.model)
                payload = {k: v for k, v in payload.items() if k != "think"}
                continue

            if resp.status_code >= 500:
                last_error = LLMError(f"Ollama returned {resp.status_code}: {resp.text[:200]}")
                logger.warning("Ollama 5xx (attempt %d/%d)", attempt, self.max_retries)
                self._sleep(attempt)
                continue

            if resp.status_code != 200:
                raise LLMError(
                    f"Ollama returned {resp.status_code} for model {self.model!r}: "
                    f"{resp.text[:300]}"
                )

            try:
                return resp.json()
            except ValueError as exc:
                raise LLMError(f"Ollama returned non-JSON envelope: {resp.text[:200]}") from exc

        raise LLMError(
            f"Ollama unreachable at {self.host} after {self.max_retries} attempts: {last_error}"
        )

    def _sleep(self, attempt: int) -> None:
        time.sleep(self.backoff ** (attempt - 1))


def build_client(project: Any) -> OllamaClient:
    """Construct an :class:`OllamaClient` from a project's config (duck-typed)."""
    return OllamaClient(
        host=project.ollama_host,
        model=project.model,
        num_ctx=project.num_ctx,
        keep_alive=project.keep_alive,
    )


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` blocks, including an unclosed leading one."""
    text = _THINK_RE.sub("", text)
    # A model may emit an opening <think> that its stop tokens never closed.
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0]
    return text


def parse_json(text: str) -> Any:
    """Parse a JSON value from noisy model output.

    Strips think tags and markdown fences, then—if the whole string still
    isn't valid JSON—extracts the first balanced ``{...}`` or ``[...]`` span.
    """
    cleaned = strip_think(text).strip()
    candidate = _strip_fences(cleaned)
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    span = _first_json_span(candidate)
    if span is None:
        raise LLMError(f"no JSON found in model output: {text[:200]!r}")
    try:
        return json.loads(span)
    except ValueError as exc:
        raise LLMError(f"model output was not valid JSON: {span[:200]!r}") from exc


def _strip_fences(text: str) -> str:
    if "```" not in text:
        return text
    lines = text.splitlines()
    kept = [ln for ln in lines if not ln.strip().startswith("```")]
    return "\n".join(kept).strip()


def _first_json_span(text: str) -> str | None:
    """Return the first balanced JSON object/array substring, or ``None``."""
    start = _first_index(text, "{", "[")
    if start is None:
        return None
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _first_index(text: str, *chars: str) -> int | None:
    positions = [text.find(c) for c in chars]
    positions = [p for p in positions if p >= 0]
    return min(positions) if positions else None


def _mentions_think(resp: requests.Response) -> bool:
    try:
        return "think" in resp.text.lower()
    except Exception:
        return False
