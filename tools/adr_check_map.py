#!/usr/bin/env python3
"""Map every ADR constraint to the check that enforces it.

Why this exists
---------------
An ADR states constraints in prose, and prose cannot stop the next change. A rule
that lives only in a document is a rule the next agent will read once, interpret
generously, and violate in a way no test notices -- which is exactly how the
"REGRESSION SNAPSHOT" labelling and the stale `+m5.2b` provenance line survived
review passes.

So each constraint names its enforcement:

    - **Cell-aligned shifts only.** ... -- enforced-by: tests/test_m54_forces.py::test_coenergy_refuses_non_aligned_delta_and_insufficient_clearance

Two failure modes this catches, and they are different:

  * **Dangling** -- the named file or test does not exist. This is a false claim in
    the documentation: worse than no link, because it reads as enforced. Fails.
  * **Untagged** -- the constraint names no check at all. Sometimes legitimate
    (the rule genuinely needs human judgement), so it is reported and counted,
    not fatal, until `--strict` is used to drive adoption down.

Usage
-----
    adr_check_map.py                  # summary; fails only on dangling links
    adr_check_map.py --strict         # also fail on untagged constraints
    adr_check_map.py --list-untagged  # print each untagged constraint
    adr_check_map.py --json

Exit codes
----------
    0  no dangling enforcement links
    1  at least one dangling link (or untagged constraints under --strict)
    2  usage error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TAG_RE = re.compile(r"enforced-by:?\s*`?([A-Za-z0-9_./\-]+(?:::[A-Za-z0-9_]+)?)`?", re.IGNORECASE)
CONSTRAINT_RE = re.compile(r"^\s*(?:[-*]\s+\*\*|\d+\.\s+\*\*)")
EXIT_USAGE = 2


def find_adrs(root: Path) -> list[Path]:
    adr_dir = root / "docs" / "adr"
    if not adr_dir.is_dir():
        return []
    return sorted(p for p in adr_dir.glob("*.md") if p.is_file())


def _resolve_ref(root: Path, ref: str) -> tuple[bool, str]:
    """Return (ok, reason) for a `path` or `path::symbol` enforcement reference."""
    path_part, _, symbol = ref.partition("::")
    target = (root / path_part).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return False, "points outside the repository"
    if not target.exists():
        return False, "file not found"
    if symbol:
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - unreadable file
            return False, f"unreadable: {exc}"
        if not re.search(rf"\b(?:def|class)\s+{re.escape(symbol)}\b", text):
            return False, f"symbol '{symbol}' not defined in {path_part}"
    return True, "ok"


def scan(root: Path) -> dict:
    total = enforced = 0
    dangling: list[dict] = []
    untagged: list[dict] = []
    per_adr: list[dict] = []

    for adr in find_adrs(root):
        lines = adr.read_text(encoding="utf-8", errors="replace").splitlines()
        a_enforced = a_total = 0
        for number, line in enumerate(lines, start=1):
            if not CONSTRAINT_RE.match(line):
                continue
            a_total += 1
            total += 1
            tag = TAG_RE.search(line)
            if not tag:
                untagged.append(
                    {
                        "adr": adr.name,
                        "line": number,
                        "text": line.strip()[:160],
                    }
                )
                continue
            ref = tag.group(1)
            ok, reason = _resolve_ref(root, ref)
            if ok:
                a_enforced += 1
                enforced += 1
            else:
                dangling.append(
                    {
                        "adr": adr.name,
                        "line": number,
                        "ref": ref,
                        "reason": reason,
                    }
                )
        per_adr.append(
            {
                "adr": adr.name,
                "constraints": a_total,
                "enforced": a_enforced,
                "untagged": a_total - a_enforced,
            }
        )

    return {
        "constraints": total,
        "enforced": enforced,
        "untagged": len(untagged) - len(dangling),
        "dangling": dangling,
        "untagged_items": untagged,
        "per_adr": per_adr,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map ADR constraints to the checks that enforce them."
    )
    parser.add_argument("--root", default=".", help="repository root (default: cwd)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--strict", action="store_true", help="fail when constraints are untagged")
    parser.add_argument(
        "--list-untagged", action="store_true", help="print each untagged constraint"
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not find_adrs(root):
        print(f"adr_check_map: no ADRs under {root / 'docs' / 'adr'}", file=sys.stderr)
        return EXIT_USAGE

    result = scan(root)

    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"ADR enforcement map — {result['constraints']} constraints across {len(result['per_adr'])} ADRs"
        )
        for row in result["per_adr"]:
            flag = "" if row["untagged"] == 0 else f"  ({row['untagged']} untagged)"
            print(f"  {row['adr']:<38} {row['enforced']}/{row['constraints']} enforced{flag}")
        print(
            f"\n  enforced {result['enforced']}/{result['constraints']}"
            f" · untagged {result['untagged']} · dangling {len(result['dangling'])}"
        )
        if result["dangling"]:
            print(
                "\nDANGLING enforcement links — the documentation claims a check that does not exist:"
            )
            for item in result["dangling"]:
                print(f"  {item['adr']}:{item['line']}  {item['ref']}  -> {item['reason']}")
        if args.list_untagged and result["untagged_items"]:
            print("\nUntagged constraints (prose only — review or tag):")
            for item in result["untagged_items"]:
                print(f"  {item['adr']}:{item['line']}  {item['text']}")
        elif result["untagged"]:
            print("\n  (use --list-untagged to see which constraints rely on prose alone)")

    if result["dangling"]:
        return 1
    if args.strict and result["untagged"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
