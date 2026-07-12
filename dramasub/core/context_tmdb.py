"""TMDB API client and context assembly.

TMDB is the primary online context source: it supports localized queries, so
we fetch the series overview, cast, and episode synopses in **both** the
source and target languages. Everything is cached under ``<project>/cache/``;
once cached, :func:`build_series_context` and :func:`build_episode_synopsis`
work fully offline.

The API key comes from the ``TMDB_API_KEY`` environment variable or the
project config — never hardcoded or committed. Both a v3 API key (query param)
and a v4 read access token (Bearer header) are supported.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from dramasub.core import _cache
from dramasub.core.errors import ContextFetchError

logger = logging.getLogger(__name__)

TMDB_BASE = "https://api.themoviedb.org/3"
_MAX_CAST = 12


def resolve_api_key(project: Any = None) -> str | None:
    """Return the TMDB key from the environment or project config."""
    key = os.environ.get("TMDB_API_KEY")
    if key:
        return key.strip()
    if project is not None:
        cfg = getattr(project, "config", {}) or {}
        value = cfg.get("tmdb_api_key")
        if value:
            return str(value).strip()
    return None


@dataclass
class TMDBClient:
    """Thin wrapper over the TMDB v3 REST API."""

    api_key: str
    base_url: str = TMDB_BASE
    timeout: float = 30.0
    session: requests.Session = field(default_factory=requests.Session, repr=False)

    def _get(self, path: str, language: str) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        params = {"language": language}
        headers = {"accept": "application/json"}
        if self.api_key.startswith("eyJ"):  # v4 read access token (JWT)
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            params["api_key"] = self.api_key
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ContextFetchError(f"TMDB request failed: {exc}") from exc
        if resp.status_code != 200:
            raise ContextFetchError(f"TMDB {path} returned {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ContextFetchError(f"TMDB returned non-JSON for {path}") from exc

    def tv_details(self, tmdb_id: int, language: str) -> dict[str, Any]:
        return self._get(f"tv/{tmdb_id}", language)

    def tv_credits(self, tmdb_id: int, language: str) -> dict[str, Any]:
        return self._get(f"tv/{tmdb_id}/aggregate_credits", language)

    def tv_season(self, tmdb_id: int, season: int, language: str) -> dict[str, Any]:
        return self._get(f"tv/{tmdb_id}/season/{season}", language)


def fetch_series_context(
    project: Any,
    client: TMDBClient,
    *,
    season: int = 1,
    refresh: bool = False,
) -> dict[str, str]:
    """Fetch and cache series details, cast, and a season pack in both languages.

    Returns a map of cache-key -> file path for what was written/reused.
    """
    if project.tmdb_id is None:
        raise ContextFetchError(
            "project has no tmdb_id; set it in project.yaml before fetching TMDB context"
        )
    tmdb_id = project.tmdb_id
    written: dict[str, str] = {}
    for language in _languages(project):
        _fetch_cached(
            project, f"tmdb_tv_{tmdb_id}_{language}", refresh,
            lambda lang=language: client.tv_details(tmdb_id, lang), written,
        )
        _fetch_cached(
            project, f"tmdb_credits_{tmdb_id}_{language}", refresh,
            lambda lang=language: client.tv_credits(tmdb_id, lang), written,
        )
        _fetch_cached(
            project, f"tmdb_season_{tmdb_id}_s{season}_{language}", refresh,
            lambda lang=language: client.tv_season(tmdb_id, season, lang), written,
        )
    logger.info("TMDB context cached for tmdb_id=%s (%d files)", tmdb_id, len(written))
    return written


def build_series_context(project: Any) -> str:
    """Compose a series-context block from cached TMDB data (offline)."""
    tmdb_id = project.tmdb_id
    if tmdb_id is None:
        return ""
    src, tgt = project.source_language, project.target_language
    details_src = _read(project, f"tmdb_tv_{tmdb_id}_{src}")
    details_tgt = _read(project, f"tmdb_tv_{tmdb_id}_{tgt}")
    if not details_src and not details_tgt:
        return ""

    lines: list[str] = []
    name = (details_tgt or {}).get("name") or (details_src or {}).get("name")
    original = (details_src or details_tgt or {}).get("original_name")
    if name:
        lines.append(f"Series: {name}" + (f" (original: {original})" if original and original != name else ""))
    for label, details in ((tgt, details_tgt), (src, details_src)):
        overview = (details or {}).get("overview")
        if overview:
            lines.append(f"Overview [{label}]: {overview}")

    cast = _merge_cast(
        _read(project, f"tmdb_credits_{tmdb_id}_{src}"),
        _read(project, f"tmdb_credits_{tmdb_id}_{tgt}"),
    )
    if cast:
        lines.append("Main cast (actor — character):")
        lines.extend(f"- {c}" for c in cast)
    return "\n".join(lines)


def build_episode_synopsis(project: Any, season: int, episode: int) -> str:
    """Compose an episode synopsis from cached season packs (offline)."""
    tmdb_id = project.tmdb_id
    if tmdb_id is None:
        return ""
    src, tgt = project.source_language, project.target_language
    lines: list[str] = []
    for label in (tgt, src):
        pack = _read(project, f"tmdb_season_{tmdb_id}_s{season}_{label}")
        ep = _find_episode(pack, episode)
        if ep and ep.get("overview"):
            title = ep.get("name", "")
            prefix = f"Ep {episode}" + (f" \"{title}\"" if title else "")
            lines.append(f"{prefix} [{label}]: {ep['overview']}")
    return "\n".join(lines)


# -- internals -------------------------------------------------------------
def _languages(project: Any) -> list[str]:
    langs = []
    for lang in (project.source_language, project.target_language):
        if lang not in langs:
            langs.append(lang)
    return langs


def _fetch_cached(project, key, refresh, fetch, written) -> None:
    path = _cache.cache_file(project.cache_dir, key)
    if refresh or not path.is_file():
        _cache.write_json(path, fetch())
    written[key] = str(path)


def _read(project: Any, key: str) -> dict[str, Any] | None:
    return _cache.read_json(_cache.cache_file(project.cache_dir, key))


def _merge_cast(
    credits_src: dict[str, Any] | None, credits_tgt: dict[str, Any] | None
) -> list[str]:
    src_cast = (credits_src or {}).get("cast", []) or []
    tgt_by_id = {c.get("id"): c for c in (credits_tgt or {}).get("cast", []) or []}
    out: list[str] = []
    for person in src_cast[:_MAX_CAST]:
        actor = person.get("name", "?")
        char_src = _character_of(person)
        char_tgt = _character_of(tgt_by_id.get(person.get("id"), {}))
        chars = char_tgt if char_tgt == char_src else " / ".join(filter(None, [char_tgt, char_src]))
        out.append(f"{actor} — {chars}" if chars else actor)
    return out


def _character_of(person: dict[str, Any]) -> str:
    roles = person.get("roles") or []
    if roles:
        return roles[0].get("character", "")
    return person.get("character", "")


def _find_episode(pack: dict[str, Any] | None, episode: int) -> dict[str, Any] | None:
    for ep in (pack or {}).get("episodes", []) or []:
        if ep.get("episode_number") == episode:
            return ep
    return None
