"""Series bible: characters, relationships, address table, glossary.

The bible is the memory that makes translations consistent across episodes.
It is stored as ``bible.yaml`` and is **hand-editable** — users correct the
address table and glossary directly. Two rules follow from that:

* **Preserve unknown keys.** We wrap the raw loaded dict rather than mapping
  it onto rigid dataclasses, so any extra keys a user adds survive round-trips.
* **Append/update only.** Code never silently deletes an entry the model
  didn't mention this episode. Every automatic change is recorded in
  ``change_log`` so the user can review and revert.

The **address table is directed**: A→B may differ from B→A. Each row records
the target-language term the speaker uses for *themself* (``self``) and for
the *listener* (``other``) — free-form strings, so the schema is
language-neutral even though the defaults document Vietnamese xưng hô.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dramasub.core._yaml import read_yaml, write_yaml

logger = logging.getLogger(__name__)

# Top-level sections, in the order they appear in a freshly written bible.
_SECTIONS = ("characters", "relationships", "address", "glossary", "change_log")


class Bible:
    """A wrapper over the raw ``bible.yaml`` mapping.

    Sections are exposed as live lists (mutating them mutates the underlying
    data). Unknown top-level keys are left untouched.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = data if data is not None else {}
        for section in _SECTIONS:
            self.data.setdefault(section, [])

    # -- section accessors -------------------------------------------------
    @property
    def characters(self) -> list[dict[str, Any]]:
        return self.data["characters"]

    @property
    def relationships(self) -> list[dict[str, Any]]:
        return self.data["relationships"]

    @property
    def address(self) -> list[dict[str, Any]]:
        return self.data["address"]

    @property
    def glossary(self) -> list[dict[str, Any]]:
        return self.data["glossary"]

    @property
    def change_log(self) -> list[dict[str, Any]]:
        return self.data["change_log"]

    # -- lookups -----------------------------------------------------------
    def character_names(self) -> set[str]:
        """Canonical names plus aliases, for matching mentions in dialogue."""
        names: set[str] = set()
        for char in self.characters:
            name = char.get("name")
            if name:
                names.add(name)
            for alias in char.get("aliases", []) or []:
                names.add(alias)
        return names

    def characters_for(self, names: set[str]) -> list[dict[str, Any]]:
        """Character entries whose name or an alias is in *names*."""
        result = []
        for char in self.characters:
            candidates = {char.get("name")} | set(char.get("aliases", []) or [])
            if candidates & names:
                result.append(char)
        return result

    def name_renderings(self, names: set[str]) -> list[tuple[str, str]]:
        """``(name, target)`` pairs for present characters with a frozen rendering.

        Injected into every pass-2 chunk so a character's target-language name
        stays byte-identical everywhere it appears.
        """
        out: list[tuple[str, str]] = []
        for char in self.characters_for(names):
            target = char.get("target")
            if char.get("name") and target:
                out.append((char["name"], target))
        return out

    def address_rows_for(self, names: set[str]) -> list[dict[str, Any]]:
        """Directed address rows where both ``from`` and ``to`` are in *names*.

        Never dump the whole table into a prompt — pass only the pairs present
        in the chunk (see AGENTS.md "Chunking and context windows").
        """
        return [
            row
            for row in self.address
            if row.get("from") in names and row.get("to") in names
        ]

    # -- mutations (append/update only) ------------------------------------
    def add_or_update_character(
        self,
        name: str,
        *,
        target: str | None = None,
        role: str | None = None,
        aliases: list[str] | None = None,
        note: str | None = None,
    ) -> str | None:
        """Add a character or fill in missing fields. Returns a change note.

        ``target`` is the frozen target-language rendering of the name. Once
        set it is never overwritten automatically (only filled if absent), so
        a name can't drift between episodes — the user hand-edits to change it.
        """
        if not name:
            return None
        existing = _find(self.characters, name=name)
        if existing is None:
            entry: dict[str, Any] = {"name": name}
            if target:
                entry["target"] = target
            if role:
                entry["role"] = role
            if aliases:
                entry["aliases"] = list(aliases)
            if note:
                entry["note"] = note
            self.characters.append(entry)
            rendered = f" = {target!r}" if target else ""
            return f"added character {name!r}{rendered}"
        return _fill_missing(
            existing, target=target, role=role, aliases=aliases, note=note, label=name
        )

    def add_or_update_relationship(
        self, frm: str, to: str, *, kind: str | None = None, note: str | None = None
    ) -> str | None:
        if not frm or not to:
            return None
        existing = _find(self.relationships, **{"from": frm, "to": to})
        if existing is None:
            entry: dict[str, Any] = {"from": frm, "to": to}
            if kind:
                entry["type"] = kind
            if note:
                entry["note"] = note
            self.relationships.append(entry)
            return f"added relationship {frm!r}->{to!r}"
        return _fill_missing(existing, type=kind, note=note, label=f"{frm}->{to}")

    def add_or_update_address(
        self,
        frm: str,
        to: str,
        *,
        self_term: str,
        other: str,
        since_episode: int | None = None,
        note: str | None = None,
    ) -> str | None:
        """Add or update a directed address-table row.

        An update to ``self``/``other`` (a register change) overwrites the
        row's terms but is logged so the user can revert.
        """
        if not frm or not to:
            return None
        existing = _find(self.address, **{"from": frm, "to": to})
        if existing is None:
            entry: dict[str, Any] = {
                "from": frm,
                "to": to,
                "self": self_term,
                "other": other,
            }
            if since_episode is not None:
                entry["since_episode"] = since_episode
            if note:
                entry["note"] = note
            self.address.append(entry)
            return f"added address {frm!r}->{to!r} (self={self_term}, other={other})"
        changed = []
        if self_term and existing.get("self") != self_term:
            changed.append(f"self {existing.get('self')!r}->{self_term!r}")
            existing["self"] = self_term
        if other and existing.get("other") != other:
            changed.append(f"other {existing.get('other')!r}->{other!r}")
            existing["other"] = other
        if since_episode is not None and existing.get("since_episode") != since_episode:
            existing["since_episode"] = since_episode
        if note:
            existing["note"] = note
        if changed:
            return f"updated address {frm!r}->{to!r}: " + ", ".join(changed)
        return None

    def add_glossary(self, source: str, target: str, *, note: str | None = None) -> str | None:
        if not source or not target:
            return None
        existing = _find(self.glossary, source=source)
        if existing is None:
            entry: dict[str, Any] = {"source": source, "target": target}
            if note:
                entry["note"] = note
            self.glossary.append(entry)
            return f"added glossary {source!r}={target!r}"
        if existing.get("target") != target:
            old = existing.get("target")
            existing["target"] = target
            if note:
                existing["note"] = note
            return f"updated glossary {source!r}: {old!r}->{target!r}"
        return None

    def log_change(self, episode: int, kind: str, detail: str) -> None:
        self.change_log.append({"episode": episode, "kind": kind, "detail": detail})

    def apply_updates(self, episode: int, updates: dict[str, Any]) -> list[str]:
        """Apply pass-1 proposed updates. Returns the list of change notes.

        ``updates`` mirrors the ``proposed_updates`` block of an episode's
        ``context.yaml``. Every applied change is appended to ``change_log``.
        """
        notes: list[str] = []

        for char in updates.get("characters", []) or []:
            note = self.add_or_update_character(
                char.get("name", ""),
                target=char.get("target"),
                role=char.get("role"),
                aliases=char.get("aliases"),
                note=char.get("note"),
            )
            _record(self, notes, episode, "character", note)

        for rel in updates.get("relationships", []) or []:
            note = self.add_or_update_relationship(
                rel.get("from", ""),
                rel.get("to", ""),
                kind=rel.get("type"),
                note=rel.get("note"),
            )
            _record(self, notes, episode, "relationship", note)

        for addr in updates.get("address", []) or []:
            note = self.add_or_update_address(
                addr.get("from", ""),
                addr.get("to", ""),
                self_term=addr.get("self", ""),
                other=addr.get("other", ""),
                since_episode=addr.get("since_episode", episode),
                note=addr.get("note"),
            )
            _record(self, notes, episode, "address", note)

        for term in updates.get("glossary", []) or []:
            note = self.add_glossary(
                term.get("source", ""), term.get("target", ""), note=term.get("note")
            )
            _record(self, notes, episode, "glossary", note)

        return notes


def new_bible() -> Bible:
    """An empty bible with all sections present."""
    return Bible({})


def load_bible(path: str | os.PathLike[str]) -> Bible:
    return Bible(read_yaml(path))


def save_bible(bible: Bible, path: str | os.PathLike[str]) -> None:
    write_yaml(path, bible.data)


def _find(rows: list[dict[str, Any]], **match: Any) -> dict[str, Any] | None:
    for row in rows:
        if all(row.get(k) == v for k, v in match.items()):
            return row
    return None


def _fill_missing(entry: dict[str, Any], *, label: str, **fields: Any) -> str | None:
    """Fill only fields that are currently absent/empty. Returns a note."""
    filled = []
    for key, value in fields.items():
        if value and not entry.get(key):
            entry[key] = list(value) if isinstance(value, list) else value
            filled.append(key)
    if filled:
        return f"updated {label!r}: filled {', '.join(filled)}"
    return None


def _record(
    bible: Bible, notes: list[str], episode: int, kind: str, note: str | None
) -> None:
    if note:
        notes.append(note)
        bible.log_change(episode, kind, note)
        logger.info("bible update (ep %d, %s): %s", episode, kind, note)
