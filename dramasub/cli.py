"""Command-line interface for dramasub.

This is the only module that reads argv, prints, and configures logging.
Everything else lives in :mod:`dramasub.core` and raises typed
:class:`~dramasub.core.errors.DramasubError`s, which are caught here and turned
into a friendly message plus a non-zero exit code.

Commands::

    init      create a project directory
    context   fetch online context (TMDB / Wikipedia) into the cache
    bible     show the series bible
    translate run the two-pass pipeline on one episode
    qc        re-run quality checks on an already-translated episode
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dramasub.core import (
    context_tmdb,
    context_wiki,
    llm,
    pass1,
    pass2,
    project as project_mod,
    qc,
    subtitle,
)
from dramasub.core.errors import DramasubError


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", 0))
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except DramasubError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


# -- commands --------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    project = project_mod.create(
        args.directory,
        title=args.title,
        tmdb_id=args.tmdb_id,
        source_language=args.source,
        target_language=args.target,
        model=args.model,
    )
    print(f"created project {project.title!r} at {project.root}")
    print(f"  languages: {project.source_language} -> {project.target_language}")
    print(f"  model: {project.model}")
    print(f"  edit {project.project_yaml} and {project.bible_yaml} as needed")
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    project = project_mod.load(args.project)
    did_something = False

    if not args.no_tmdb and project.tmdb_id is not None:
        api_key = context_tmdb.resolve_api_key(project)
        if not api_key:
            print(
                "warning: no TMDB API key (set TMDB_API_KEY or tmdb_api_key in "
                "project.yaml); skipping TMDB",
                file=sys.stderr,
            )
        else:
            client = context_tmdb.TMDBClient(api_key)
            written = context_tmdb.fetch_series_context(
                project, client, season=args.season, refresh=args.refresh
            )
            print(f"fetched TMDB context ({len(written)} files cached)")
            did_something = True
    elif not args.no_tmdb:
        print("note: no tmdb_id in project.yaml; skipping TMDB", file=sys.stderr)

    titles = {
        project.source_language: args.wiki_source,
        project.target_language: args.wiki_target,
    }
    if any(titles.values()):
        written = context_wiki.fetch_series_wiki(project, titles, refresh=args.refresh)
        print(f"fetched Wikipedia context ({len(written)} editions cached)")
        did_something = True

    if not did_something:
        print("nothing fetched; pass --wiki-source/--wiki-target or set tmdb_id")
    return 0


def cmd_bible(args: argparse.Namespace) -> int:
    project = project_mod.load(args.project)
    bible = project.load_bible()
    print(f"bible for {project.title!r} — edit {project.bible_yaml}")
    print(f"\ncharacters ({len(bible.characters)}):")
    for char in bible.characters:
        aliases = f" (aka {', '.join(char['aliases'])})" if char.get("aliases") else ""
        role = f" — {char['role']}" if char.get("role") else ""
        print(f"  - {char.get('name', '?')}{role}{aliases}")
    print(f"\naddress table ({len(bible.address)}):")
    for row in bible.address:
        since = f" since ep {row['since_episode']}" if row.get("since_episode") is not None else ""
        print(f"  - {row.get('from')} -> {row.get('to')}: "
              f"self={row.get('self')}, other={row.get('other')}{since}")
    print(f"\nglossary ({len(bible.glossary)}):")
    for term in bible.glossary:
        note = f"  # {term['note']}" if term.get("note") else ""
        print(f"  - {term.get('source')} = {term.get('target')}{note}")
    if bible.change_log:
        print(f"\nrecent bible changes ({len(bible.change_log)}):")
        for entry in bible.change_log[-10:]:
            print(f"  - ep{entry.get('episode')} [{entry.get('kind')}] {entry.get('detail')}")
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    project = project_mod.load(args.project)
    if args.direct and args.pass2_only:
        raise DramasubError("--direct and --pass2-only are mutually exclusive")
    client = llm.build_client(project)

    if args.file is not None:
        print(f"importing episode {args.episode} source: {args.file}")
        project.import_episode_source(args.episode, args.file)
    src_path = project.episode_source_existing(args.episode)
    if src_path is None:
        raise DramasubError(
            f"no source imported for episode {args.episode}; pass the subtitle file"
        )
    doc = subtitle.load(src_path)
    bible = project.load_bible()

    series_context = _series_context(project, args.season, args.episode)
    prev_summary = _prev_summary(project, args.episode)

    episode_context = None
    if args.pass2_only:
        # Reuse the saved pass-1 context and current bible — for re-translating
        # after a prompt/model/bible change without re-running the analysis.
        episode_context = project.load_episode_context(args.episode)
        if episode_context is None:
            raise DramasubError(
                f"no saved context for episode {args.episode}; run a full "
                "translate first (or use --direct)"
            )
        print(f"pass 2 only: reusing saved context "
              f"({len(episode_context.get('characters_present', []))} characters)")
    elif args.direct:
        # One-pass direct translate: no pass-1 analysis, no bible updates —
        # only whatever online (TMDB) context exists. Lower quality/consistency
        # but roughly half the model calls; useful as a fast baseline.
        print("direct mode: one pass, no context extraction (TMDB context only)")
    else:
        print(f"pass 1: extracting context from {len(doc.translatable_indices())} cues "
              f"(model {project.model})...")
        p1 = pass1.extract_context(
            project, bible, doc,
            episode=args.episode, llm=client,
            series_context=series_context, prev_summary=prev_summary,
        )
        project.save_bible(bible)
        episode_context = p1.context
        print(f"  characters present: {', '.join(p1.characters_present) or '(none identified)'}")
        if p1.applied_updates:
            print(f"  applied {len(p1.applied_updates)} bible update(s):")
            for note in p1.applied_updates:
                print(f"    - {note}")

    print("translating (one pass)..." if args.direct else "pass 2: translating...")
    result = pass2.translate_episode(
        project, bible, doc,
        episode=args.episode, llm=client,
        episode_context=episode_context,
        series_context=series_context, prev_summary=prev_summary,
        generate_summary=not args.no_summary and not args.direct,
    )
    _report(result)
    return 1 if result.failed_indices else 0


def cmd_qc(args: argparse.Namespace) -> int:
    project = project_mod.load(args.project)
    src_path = project.episode_source_existing(args.episode)
    out_path = project.episode_output_existing(args.episode)
    if src_path is None:
        raise DramasubError(f"no source for episode {args.episode}; run translate first")
    if out_path is None:
        raise DramasubError(f"no output for episode {args.episode}; run translate first")
    source = subtitle.load(src_path)
    output = subtitle.load(out_path)
    subtitle.verify_timing(source, output)
    bible = project.load_bible()
    warnings = qc.run_qc(
        bible,
        source,
        output,
        max_line_chars=project.max_line_chars,
        target_language=project.target_language,
    )
    print(f"QC for episode {args.episode}: {len(warnings)} warning(s)")
    for warning in warnings:
        print(f"  {warning}")
    return 1 if warnings else 0


# -- helpers ---------------------------------------------------------------
def _series_context(project: project_mod.Project, season: int, episode: int) -> str:
    parts = [
        context_tmdb.build_series_context(project),
        context_tmdb.build_episode_synopsis(project, season, episode),
        context_wiki.build_wiki_context(project),
    ]
    return "\n".join(p for p in parts if p)


def _prev_summary(project: project_mod.Project, episode: int) -> str:
    if episode <= 1:
        return ""
    prev = project.episode_summary(episode - 1)
    return prev.read_text(encoding="utf-8").strip() if prev.is_file() else ""


def _report(result: pass2.TranslateResult) -> None:
    print("\n=== translation summary ===")
    print(f"  cues translated: {result.translated_count}/{result.total_cues}")
    if result.output_path:
        print(f"  output: {result.output_path}")
    if result.failed_indices:
        print(f"  FAILED cues ({len(result.failed_indices)}): {result.failed_indices}")
    if result.qc_warnings:
        print(f"  QC warnings ({len(result.qc_warnings)}):")
        for warning in result.qc_warnings[:20]:
            print(f"    {warning}")
        if len(result.qc_warnings) > 20:
            print(f"    ... and {len(result.qc_warnings) - 20} more")
    else:
        print("  QC warnings: none")
    if result.summary:
        print("  episode summary written")


def _load_dotenv(path: Path | None = None) -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` in the working directory.

    Populates ``os.environ`` (existing environment always wins), so settings
    like ``TMDB_API_KEY`` and ``OLLAMA_HOST`` can live in a local, gitignored
    ``.env`` instead of being exported by hand. Absent file → no-op.
    """
    path = path or Path(".env")
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dramasub", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for progress, -vv for debug logging")
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="create a new project directory")
    p_init.add_argument("directory", type=Path)
    p_init.add_argument("--title", required=True)
    p_init.add_argument("--tmdb-id", type=int, default=None, dest="tmdb_id")
    p_init.add_argument("--source", default="ko", help="source language code (default: ko)")
    p_init.add_argument("--target", default="vi", help="target language code (default: vi)")
    p_init.add_argument("--model", default=None, help="override the default Ollama model")
    p_init.set_defaults(func=cmd_init)

    p_ctx = sub.add_parser("context", help="fetch online context into the cache")
    p_ctx.add_argument("project", type=Path)
    p_ctx.add_argument("--season", type=int, default=1)
    p_ctx.add_argument("--refresh", action="store_true", help="re-fetch even if cached")
    p_ctx.add_argument("--no-tmdb", action="store_true", help="skip TMDB")
    p_ctx.add_argument("--wiki-source", default=None, help="source-edition Wikipedia article title")
    p_ctx.add_argument("--wiki-target", default=None, help="target-edition Wikipedia article title")
    p_ctx.set_defaults(func=cmd_context)

    p_bible = sub.add_parser("bible", help="show the series bible")
    p_bible.add_argument("project", type=Path)
    p_bible.set_defaults(func=cmd_bible)

    p_tr = sub.add_parser("translate", help="translate one episode (two-pass)")
    p_tr.add_argument("project", type=Path)
    p_tr.add_argument("--episode", type=int, required=True)
    p_tr.add_argument("--season", type=int, default=1)
    p_tr.add_argument("file", type=Path, nargs="?", default=None,
                      help="the source subtitle file (optional if already imported)")
    p_tr.add_argument("--no-summary", action="store_true", help="skip episode summary")
    p_tr.add_argument("--direct", action="store_true",
                      help="one-pass direct translate: skip pass 1 and the bible, "
                           "use only online (TMDB) context (faster, less consistent)")
    p_tr.add_argument("--pass2-only", action="store_true", dest="pass2_only",
                      help="skip pass 1 and reuse the episode's saved context.yaml "
                           "and current bible (for re-translating after a prompt, "
                           "model, or bible change)")
    p_tr.set_defaults(func=cmd_translate)

    p_qc = sub.add_parser("qc", help="re-run QC on a translated episode")
    p_qc.add_argument("project", type=Path)
    p_qc.add_argument("--episode", type=int, required=True)
    p_qc.set_defaults(func=cmd_qc)

    return parser


if __name__ == "__main__":
    sys.exit(main())
