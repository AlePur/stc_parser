"""
STC Compiler
============
CLI entry point: parse → lint → (optionally) emit Mermaid diagram.

Usage:
    python stc_compiler.py <file.stc> [--no-diagram] [--out <output.md>]

Exit codes:
    0  Success (no errors; warnings may be present)
    1  Parse error or lint errors found
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from stc_ast import ServiceAST, Routine, State
from stc_parser import parse_file, ParseError
from stc_linter import STCLinter, Diagnostic


# ---------------------------------------------------------------------------
# Mermaid diagram generator
# ---------------------------------------------------------------------------

def _mermaid_state_description(state: State, routines_map: dict) -> str:
    """
    Build a single-line description for a Mermaid state box.
    Returns only the description part (caller prepends 'StateName : ').
    """
    parts = []
    for step in state.steps:
        r = routines_map.get(step)
        if r and r.lock_target:
            parts.append(f"{step}[locked]")
        else:
            parts.append(step)

    return "steps(" + ", ".join(parts) + ")"


def generate_mermaid(ast: ServiceAST) -> str:
    """Generate a Mermaid stateDiagram-v2 string from the AST."""
    lines: List[str] = ["```mermaid", "stateDiagram-v2"]

    routine_map: dict = {r.name: r for r in ast.routines}

    # State labels (only emit a custom label if there are steps)
    for state in ast.states:
        if state.steps:
            desc = _mermaid_state_description(state, routine_map)
            lines.append(f'    {state.name} : {desc}')

    lines.append("")

    # Entry arrow from [*] to first state
    for state in ast.states:
        if state.is_first:
            lines.append(f"    [*] --> {state.name}")

    # All transitions
    for state in ast.states:
        for trans in state.transitions:
            lines.append(f"    {state.name} --> {trans.target_state} : {trans.event}")

    # Terminal states back to [*]
    for state in ast.states:
        if state.is_terminal:
            lines.append(f"    {state.name} --> [*]")

    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
RED    = "\033[31m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
DIM    = "\033[2m"


def _colored(text: str, color: str) -> str:
    """Wrap text in ANSI escape codes if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{color}{text}{RESET}"
    return text


def _print_header(text: str):
    print(_colored(f"\n{'─' * 60}", DIM))
    print(_colored(f"  {text}", BOLD))
    print(_colored(f"{'─' * 60}", DIM))


def _print_diagnostics(diagnostics: List[Diagnostic], is_error: bool):
    color = RED if is_error else YELLOW
    for d in diagnostics:
        loc = _colored(f"(line {d.line})", DIM) if d.line else ""
        code = _colored(f"[{d.code}]", color + BOLD)
        kind = _colored("ERROR  " if is_error else "WARNING", color)
        print(f"  {code} {kind} {loc}")
        print(f"         {d.message}")


def _print_ast_summary(ast: ServiceAST):
    _print_header(f"Service: {ast.service_name}")

    print(_colored("  Externals:", CYAN))
    for e in ast.externals:
        print(f"    {e.type.value:<6} {e.name}")

    print(_colored("  Routines:", CYAN))
    for r in ast.routines:
        deps = ", ".join(r.dependencies) or "—"
        lock = f"  🔒 locks {r.lock_target}" if r.lock_target else ""
        rb   = "  ↩ has rollback" if r.rollback else ""
        print(f"    {r.name}({deps}){lock}{rb}")

    print(_colored("  States:", CYAN))
    for s in ast.states:
        tags = []
        if s.is_first:    tags.append("first")
        if s.is_terminal: tags.append("terminal")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        trans_str = "  →  " + ", ".join(
            f"{t.event}→{t.target_state}" for t in s.transitions
        ) if s.transitions else ""
        steps_str = f"  steps: {s.steps}" if s.steps else ""
        print(f"    {s.name}{tag_str}{steps_str}{trans_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="stc_compiler",
        description="Compile a .stc file: parse, lint, and emit a Mermaid diagram.",
    )
    parser.add_argument("file", help="Path to the .stc source file")
    parser.add_argument(
        "--no-diagram", action="store_true",
        help="Skip Mermaid diagram generation",
    )
    parser.add_argument(
        "--out", metavar="FILE",
        help="Write Mermaid diagram to this file (default: <name>.md)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the AST summary; only show lint output",
    )
    args = parser.parse_args()

    source_path = Path(args.file)
    if not source_path.exists():
        print(_colored(f"Error: file not found: {args.file}", RED), file=sys.stderr)
        sys.exit(1)

    # ── 1. Parse ─────────────────────────────────────────────────────────────
    print(_colored(f"\n⚙  Parsing {source_path.name} …", BOLD))
    try:
        ast = parse_file(str(source_path))
    except ParseError as e:
        print(_colored(f"\n✗  Parse error: {e}", RED))
        sys.exit(1)

    if not args.quiet:
        _print_ast_summary(ast)

    # ── 2. Lint ───────────────────────────────────────────────────────────────
    print(_colored("\n⚙  Linting …", BOLD))
    linter = STCLinter(ast)
    errors, warnings = linter.lint()

    if warnings:
        _print_header(f"Warnings ({len(warnings)})")
        _print_diagnostics(warnings, is_error=False)

    if errors:
        _print_header(f"Errors ({len(errors)})")
        _print_diagnostics(errors, is_error=True)
        print(_colored(f"\n✗  {len(errors)} error(s) found. Fix them before generating output.\n", RED))
        sys.exit(1)

    if not errors and not warnings:
        print(_colored("  ✓  No issues found.", GREEN))
    elif not errors:
        print(_colored(f"\n  ✓  Lint passed with {len(warnings)} warning(s).", YELLOW))

    # ── 3. Mermaid diagram ────────────────────────────────────────────────────
    if args.no_diagram:
        sys.exit(0)

    diagram = generate_mermaid(ast)

    out_path = Path(args.out) if args.out else source_path.with_suffix(".md")
    out_path.write_text(
        f"# {ast.service_name}\n\n"
        f"_Auto-generated from `{source_path.name}` by the STC compiler._\n\n"
        + diagram + "\n",
        encoding="utf-8",
    )

    print(_colored(f"\n✓  Diagram written to {out_path}\n", GREEN))
    print(diagram)
    print()


if __name__ == "__main__":
    main()
