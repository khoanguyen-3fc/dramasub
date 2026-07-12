"""Wikipedia REST API client and context assembly.

Wikipedia complements TMDB: the source-language edition gives accurate native
names and relationships, while the target-language edition gives the
community-standard name renderings. Summaries are cached under
``<project>/cache/`` and reused offline.

Uses the page-summary REST endpoint:
``https://<lang>.wikipedia.org/api/rest_v1/page/summary/<title>``.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

import requests

from dramasub.core import _cache
from dramasub.core.errors import ContextFetchError

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
# REST v1 asks clients to send a descriptive User-Agent.
_HEADERS = {"User-Agent": "dramasub/0.1 (subtitle localization tool)"}


def fetch_summary(
    title: str,
    lang: str,
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch a single page summary from the *lang* Wikipedia edition."""
    quoted = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quoted}"
    sess = session or requests
    try:
        resp = sess.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise ContextFetchError(f"Wikipedia request failed for {title!r}: {exc}") from exc
    if resp.status_code == 404:
        raise ContextFetchError(f"Wikipedia has no {lang} page for {title!r}")
    if resp.status_code != 200:
        raise ContextFetchError(
            f"Wikipedia returned {resp.status_code} for {title!r} ({lang})"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise ContextFetchError(f"Wikipedia returned non-JSON for {title!r}") from exc


def fetch_series_wiki(
    project: Any,
    titles: dict[str, str],
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> dict[str, str]:
    """Fetch and cache summaries for the given per-language page titles.

    *titles* maps a language code to the article title in that edition, e.g.
    ``{"ko": "내일 봐요 사장님", "vi": "..."}``. Missing pages are skipped with
    a warning rather than aborting the whole fetch.
    """
    written: dict[str, str] = {}
    for lang, title in titles.items():
        if not title:
            continue
        key = f"wiki_{lang}"
        path = _cache.cache_file(project.cache_dir, key)
        if not refresh and path.is_file():
            written[key] = str(path)
            continue
        try:
            data = fetch_summary(title, lang, session=session)
        except ContextFetchError as exc:
            logger.warning("skipping Wikipedia %s/%s: %s", lang, title, exc)
            continue
        _cache.write_json(path, data)
        written[key] = str(path)
    logger.info("Wikipedia context cached (%d editions)", len(written))
    return written


def build_wiki_context(project: Any) -> str:
    """Compose a context block from cached Wikipedia summaries (offline)."""
    lines: list[str] = []
    for lang in _languages(project):
        data = _cache.read_json(_cache.cache_file(project.cache_dir, f"wiki_{lang}"))
        extract = (data or {}).get("extract")
        if extract:
            title = (data or {}).get("title", "")
            lines.append(f"Wikipedia [{lang}] {title}: {extract}")
    return "\n".join(lines)


def _languages(project: Any) -> list[str]:
    langs = []
    for lang in (project.source_language, project.target_language):
        if lang not in langs:
            langs.append(lang)
    return langs
