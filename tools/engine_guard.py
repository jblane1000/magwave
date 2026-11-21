#!/usr/bin/env python3
"""Refuse to start a second implementation engine in the same repository.

Why this exists
---------------
An agent that ends a turn with the engine still running, or starts a second engine
because the first "seemed" finished, loses work and burns quota. That failure has
recurred: three lost round-trips and one engine killed mid-run while it was
rewriting the same files a verification pass was reading. Each time it was fixed by
writing a stronger rule in prose, and prose did not stop the next occurrence.

Prose is the weakest layer: it is read once, at the top of a turn, and it cannot
intervene at the moment of the mistake. This is the deterministic replacement --
a check that runs exactly when someone is about to launch an engine, and refuses
if another one already owns this repository.

Scope is deliberately per-repository, not per-host: two engines on two different
clones is legitimate (a bake-off does exactly that). Two engines on *one* tree is
never legitimate, because both write the same files and each invalidates the
other's verification.

Usage
-----
    engine_guard.py                     # check the current git repository
    engine_guard.py --repo /path/to/repo
    engine_guard.py --json              # machine-readable
    engine_guard.py -- agy-work --model ... --print "..."   # guard, then run
    engine_guard.py --force -- <cmd>    # override, loudly (do not use casually)

Exit codes
----------
    0  clear -- no engine owns this repository, safe to start
    2  usage error
    3  blocked -- an engine is already running against this repository
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ENGINE_BASENAMES = ("agy", "agy-work", "antigravity")
EXIT_CLEAR = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 3


def _proc_cmdline(pid: int) -> list[str] | None:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    if not raw:
        return None
    parts = [p.decode("utf-8", "replace") for p in raw.split(b"\0")]
    return [p for p in parts if p]


def _proc_cwd(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None


def _proc_state(pid: int) -> str | None:
    """Process state letter from /proc/<pid>/stat (Z = zombie, X = dead)."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
    except OSError:
        return None
    try:
        return data.rsplit(")", 1)[1].split()[0]
    except IndexError:
        return None


def _proc_elapsed_s(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            fields = fh.read().rsplit(")", 1)[1].split()
        start_ticks = float(fields[19])  # field 22 overall, 20th after comm
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
        hz = os.sysconf("SC_CLK_TCK")
        return max(0.0, uptime - start_ticks / hz)
    except (OSError, IndexError, ValueError):
        return None


def _is_engine(cmdline: list[str]) -> bool:
    for arg in cmdline:
        base = os.path.basename(arg)
        if base in ENGINE_BASENAMES:
            return True
    return False


def _targets_repo(pid: int, cmdline: list[str], repo: str) -> str | None:
    """How this process touches `repo`, or None.

    Two independent signals, because both have been observed in real incidents:
      cwd     -- the engine was launched from inside the tree;
      cmdline -- the prompt embedded the absolute repo path.
    """
    cwd = _proc_cwd(pid)
    if cwd:
        real = os.path.realpath(cwd)
        if real == repo or real.startswith(repo + os.sep):
            return "cwd"
    joined = " ".join(cmdline)
    if repo in joined:
        return "cmdline"
    return None


def find_engines(repo: str) -> list[dict]:
    """Every live engine process that owns `repo`."""
    found: list[dict] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == os.getpid():
            continue
        state = _proc_state(pid)
        if state in ("Z", "X"):
            continue
        cmdline = _proc_cmdline(pid)
        if not cmdline or not _is_engine(cmdline):
            continue
        matched = _targets_repo(pid, cmdline, repo)
        if not matched:
            continue
        elapsed = _proc_elapsed_s(pid)
        found.append(
            {
                "pid": pid,
                "matched_by": matched,
                "elapsed_s": None if elapsed is None else round(elapsed, 1),
                "elapsed": "?" if elapsed is None else _fmt_duration(elapsed),
                "cmd": " ".join(cmdline)[:200],
            }
        )
    return sorted(found, key=lambda row: row["pid"])


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def resolve_repo(explicit: str | None) -> str:
    if explicit:
        return os.path.realpath(explicit)
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env specific
        raise SystemExit(f"engine_guard: cannot locate repository: {exc}") from exc
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit("engine_guard: not inside a git repository (pass --repo PATH)")
    return os.path.realpath(out.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refuse to start a second implementation engine in one repository."
    )
    parser.add_argument("--repo", help="repository to guard (default: current git root)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--force",
        action="store_true",
        help="proceed even if an engine holds the repository (prints a loud warning)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to exec after a clear check (use: -- cmd args...)",
    )
    args = parser.parse_args(argv)

    repo = resolve_repo(args.repo)
    running = find_engines(repo)
    command = [c for c in args.command if c != "--"]

    if args.as_json:
        print(
            json.dumps(
                {
                    "repo": repo,
                    "clear": not running,
                    "blocked": bool(running) and not args.force,
                    "engines": running,
                },
                indent=2,
            )
        )
    elif running:
        print(f"BLOCKED: {len(running)} engine process(es) already own {repo}", file=sys.stderr)
        for row in running:
            print(
                f"  pid {row['pid']}  up {row['elapsed']}  matched by {row['matched_by']}\n"
                f"      {row['cmd']}",
                file=sys.stderr,
            )
        print(
            "\nTwo engines writing one tree invalidate each other's verification and waste the\n"
            "quota both are spending. Wait for it to exit, then verify its output:\n"
            "  while kill -0 <pid> 2>/dev/null; do sleep 30; done\n"
            "If the run is genuinely stale, kill it deliberately rather than starting over it.",
            file=sys.stderr,
        )
    else:
        print(f"clear: no engine owns {repo}", file=sys.stderr)

    if running and args.force:
        print("WARNING: --force given; starting a second engine anyway.", file=sys.stderr)
    elif running:
        return EXIT_BLOCKED

    if command:
        try:
            os.execvp(command[0], command)  # noqa: S606 - launching without a shell is deliberate
        except OSError as exc:
            print(f"engine_guard: cannot exec {command[0]}: {exc}", file=sys.stderr)
            return EXIT_USAGE
    return EXIT_CLEAR


if __name__ == "__main__":
    sys.exit(main())
