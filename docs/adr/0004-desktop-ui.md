# ADR 0004 — Desktop first, browser later (if ever)

- Status: accepted
- Date: 2026-11-17

## Context

The user's stated criteria for a tool are that it **looks right** and swallows his file formats. His work
happens next to CAD on a workstation. Web deployment would add a server, a browser stack, and a WebGL
engine — three new failure domains before the first field plot.

## Decision

- **PySide6 + PyVista** desktop application; PyVista gives VTK-quality 3D with Python-level ergonomics.
- CLI and Python API are first-class and complete: the GUI is a client of the same API, never the only path.
- No web UI in v1. If a browser front end is ever wanted, it talks to the API, not to the solver directly.

## Consequences

- Fastest route to a good-looking, honest 3D result.
- Two platforms to package (Linux, Windows) instead of every browser.
- Remote use is SSH + CLI first; a hosted UI is explicitly out of scope (NG4).
