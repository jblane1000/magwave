"""Benchmark corpus, evaluators and harness (PRD FR-070).

The corpus is data (``cases.json``), the evaluation is code (``checks.py``) and the
reporting is the harness (``harness.py``). Expected values and their tolerances are
research-owned: see ``docs/research/reference_formulas.py`` and
``docs/research/validation-sources.md``.
"""

from __future__ import annotations

from .harness import (
    CORPUS_SCHEMA,
    RESULTS_SCHEMA,
    CaseResult,
    EngineCaseResult,
    all_passed,
    corpus_path,
    dump_corpus,
    engine_all_passed,
    engine_results_payload,
    engine_summary,
    evaluate,
    evaluate_with_engine,
    format_engine_table,
    format_table,
    load_corpus,
    results_payload,
    run_benchmarks,
    run_engine_benchmarks,
)

__all__ = [
    "CORPUS_SCHEMA",
    "RESULTS_SCHEMA",
    "CaseResult",
    "EngineCaseResult",
    "all_passed",
    "corpus_path",
    "dump_corpus",
    "engine_all_passed",
    "engine_results_payload",
    "engine_summary",
    "evaluate",
    "evaluate_with_engine",
    "format_engine_table",
    "format_table",
    "load_corpus",
    "results_payload",
    "run_benchmarks",
    "run_engine_benchmarks",
]
