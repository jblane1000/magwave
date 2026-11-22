# 10 — Lint rules as the review layer for engine-authored code

**Status:** accepted (2026-11-21)

## Context

Implementation code in this repository is written by an engine (`agy`), and the coordinator verifies it.
That inverts the usual review economics: a human no longer reads every line before it lands, so the checks
that run on the line matter more than the opinion formed after it. Two failure classes recurred in one week:

- **A rule enforced only by installation.** "Core stays Qt-free and OCP-free" was proven by CI building the
  core without the `gui`/CAD extras. An import violation therefore surfaced minutes later, in a different
  job, with a message about a missing module rather than about the boundary that was crossed.
- **Prose that could not stop the next change.** Recurring corrections lived in review comments and ADRs,
  both of which are read once and then not enforced.

## Decision

1. **`ruff` is the deterministic review layer**, running `check` and `format --check` in CI and in every
   acceptance gate. Rule set: `E, F, I, UP, B, SIM, ARG, RUF, TID, S`. `ARG` and `RUF` were added because
   they catch the failures an agent actually produces (dead parameters, stale `# noqa`, mis-sorted
   `__all__`); `S` is a fence for future code; `PL` and `N` are deliberately **not** enabled — magic values
   and `Bz`/`mu0` naming are the physics, not defects.
2. **The Qt and OCP fences are import bans, not CI accidents.** `flake8-tidy-imports` rejects `PySide6`
   anywhere except `src/magwave/gui.py` and `OCP` anywhere except `src/magwave/cad/**`, with a message
   naming the boundary. Boundary modules are listed explicitly; a ban is never silenced to make a
   violation pass. -- enforced-by: tests/test_lint_fence.py::test_qt_import_is_refused_in_a_core_module
3. **An exception is documented at the line, with its measurement.** `os.chmod(workdir, 0o777)` in the
   three Elmer tools carries `# noqa: S103` because this host runs userns-remapped Docker: a `0700`
   directory cannot even be `chdir`-ed into by the container, verified both with the default user and with
   `--user $(id -u):$(id -g)`. The exposure is bounded to one `TemporaryDirectory` per run.
   -- enforced-by: tests/test_lint_fence.py::test_the_repository_itself_is_lint_clean

## Consequences

- A boundary violation is refused where it is written, with a reason, instead of failing a later CI job.
- `tests/test_lint_fence.py` drives the real linter over real snippets, so the fence is proven to fire
  rather than assumed to: the banned import fails in core and passes in the module that owns it.
- Style rules that fight the domain stay off. The list is expected to grow by evidence — a correction that
  recurs twice becomes a rule — not by taste.
- Not adopted: a blanket comment ban. ADRs and `method`/`notes` strings earn their keep here; what dies is
  a justification comment standing in for a check.
