"""
STC Compiler
============
CLI entry point: parse → lint → emit Mermaid markdown → (optionally) render SVG with mmdc.

Usage:
    # Compile a single file
    python stc_compiler.py checkout.stc
    python stc_compiler.py checkout.stc --out build/

    # Compile every .stc file in a directory (recursively)
    python stc_compiler.py src/
    python stc_compiler.py src/ --out build/

    # Skip SVG rendering (keep .md)
    python stc_compiler.py src/ --no-mmdc

    # Skip diagram + SVG entirely
    python stc_compiler.py src/ --no-diagram

Exit codes:
    0  All files compiled and linted with no errors
    1  One or more parse or lint errors found
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from stc_ast import DoKind, ServiceAST, State
from stc_parser import parse_file, ParseError
from stc_linter import STCLinter, Diagnostic


# ---------------------------------------------------------------------------
# ANSI colours (only when stdout is a TTY)
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
RED    = "\033[31m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
DIM    = "\033[2m"


def _c(text: str, *codes: str) -> str:
    if sys.stdout.isatty():
        return "".join(codes) + text + RESET
    return text


def _header(text: str):
    print(_c(f"\n{'─' * 60}", DIM))
    print(_c(f"  {text}", BOLD))
    print(_c(f"{'─' * 60}", DIM))


def _print_diagnostics(diagnostics: List[Diagnostic], is_error: bool):
    color = RED if is_error else YELLOW
    for d in diagnostics:
        loc = _c(f"(line {d.line})", DIM) if d.line else ""
        code = _c(f"[{d.code}]", color, BOLD)
        kind = _c("ERROR  " if is_error else "WARNING", color)
        print(f"  {code} {kind} {loc}")
        print(f"         {d.message}")


def _print_ast_summary(ast: ServiceAST):
    _header(f"Service: {ast.service_name}")
    print(_c("  Externals:", CYAN))
    for e in ast.externals:
        print(f"    {e.type.value:<6} {e.name}")

    print(_c("  Routines:", CYAN))
    for r in ast.routines:
        deps = ", ".join(r.dependencies) or "—"
        lock_resources = [
            s.do_action.resource
            for s in r.steps
            if s.do_action.kind.value == "lock" and s.do_action.resource
        ]
        has_rb = any(s.rollback_action for s in r.steps)
        unfailing = "  ⚡ unfailing" if r.is_unfailing else ""
        lock = f"  🔒 locks {', '.join(lock_resources)}" if lock_resources else ""
        rb   = "  ↩ has rollback(s)" if has_rb else ""
        print(f"    {r.name}({deps}){unfailing}{lock}{rb}")

    print(_c("  States:", CYAN))
    for s in ast.states:
        tags = []
        if s.is_first:    tags.append("first")
        if s.is_terminal: tags.append("terminal")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        trans_str = ("  →  " + ", ".join(
            f"{t.event}→{t.target_state}" for t in s.transitions
        )) if s.transitions else ""
        steps_str = f"  steps: {s.steps}" if s.steps else ""
        print(f"    {s.name}{tag_str}{steps_str}{trans_str}")


# ---------------------------------------------------------------------------
# Mermaid diagram generator
# ---------------------------------------------------------------------------

def _state_description(state: State, routine_map: dict) -> str:
    """Single-line description for a state box: 'steps(A, B[locked])'."""
    parts = []
    for step in state.steps:
        r = routine_map.get(step)
        has_lock = r and any(
            s.do_action.kind == DoKind.LOCK for s in r.steps
        )
        parts.append(f"{step}[locked]" if has_lock else step)
    return "steps(" + ", ".join(parts) + ")"


def generate_mermaid(ast: ServiceAST) -> str:
    """Return a Mermaid stateDiagram-v2 fenced code block string."""
    lines: List[str] = ["```mermaid", "stateDiagram-v2"]
    routine_map = {r.name: r for r in ast.routines}

    for state in ast.states:
        if state.steps:
            lines.append(f"    {state.name} : {_state_description(state, routine_map)}")

    lines.append("")

    for state in ast.states:
        if state.is_first:
            lines.append(f"    [*] --> {state.name}")

    for state in ast.states:
        for trans in state.transitions:
            lines.append(f"    {state.name} --> {trans.target_state} : {trans.event}")

    for state in ast.states:
        if state.is_terminal:
            lines.append(f"    {state.name} --> [*]")

    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core compile logic for a single file
# ---------------------------------------------------------------------------

def compile_one(
    source_path: Path,
    out_dir: Optional[Path],
    quiet: bool,
    no_diagram: bool,
) -> Tuple[bool, Optional[Path]]:
    """
    Parse and lint one .stc file.

    Returns:
        (success, md_path)  — md_path is None if --no-diagram or errors occurred.
    """
    print(_c(f"\n⚙  {source_path}", BOLD))

    # Parse
    try:
        ast = parse_file(str(source_path))
    except ParseError as e:
        print(_c(f"   ✗  Parse error: {e}", RED))
        return False, None

    if not quiet:
        _print_ast_summary(ast)

    # Lint
    linter = STCLinter(ast)
    errors, warnings = linter.lint()

    if warnings:
        _header(f"Warnings ({len(warnings)})")
        _print_diagnostics(warnings, is_error=False)

    if errors:
        _header(f"Errors ({len(errors)})")
        _print_diagnostics(errors, is_error=True)
        print(_c(f"\n   ✗  {len(errors)} error(s) — skipping diagram output.\n", RED))
        return False, None

    if not warnings:
        print(_c("   ✓  No issues.", GREEN))
    else:
        print(_c(f"   ✓  Lint passed with {len(warnings)} warning(s).", YELLOW))

    if no_diagram:
        return True, None

    # Write .md
    dest_dir = out_dir if out_dir else source_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    md_path = dest_dir / source_path.with_suffix(".md").name

    diagram = generate_mermaid(ast)
    md_path.write_text(
        f"# {ast.service_name}\n\n"
        f"_Auto-generated from `{source_path.name}` by the STC compiler._\n\n"
        + diagram + "\n",
        encoding="utf-8",
    )
    print(_c(f"   📄  {md_path}", CYAN))
    return True, md_path


# ---------------------------------------------------------------------------
# mmdc SVG rendering
# ---------------------------------------------------------------------------

def render_svg(md_path: Path) -> bool:
    """Run `mmdc -i <md_path> -o <svg_path>`. Returns True on success."""
    svg_path = md_path.with_suffix(".svg")
    cmd = ["mmdc", "-i", str(md_path), "-o", str(svg_path)]
    print(_c(f"   🖼   mmdc → {svg_path}", CYAN))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(_c(f"   ✗  mmdc failed for {md_path.name}:", RED))
            if result.stderr:
                for line in result.stderr.strip().splitlines():
                    print(f"       {line}")
            return False
        return True
    except FileNotFoundError:
        print(_c("   ✗  mmdc not found. Install mermaid-cli: npm i -g @mermaid-js/mermaid-cli", RED))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        prog="stc_compiler",
        description="Compile .stc files: parse → lint → Mermaid markdown → SVG.",
    )
    ap.add_argument(
        "path",
        help="A .stc file or a directory containing .stc files.",
    )
    ap.add_argument(
        "--out", metavar="DIR",
        help="Output directory for generated .md (and .svg) files. "
             "For a single file, defaults to the same directory as the source. "
             "For a directory, defaults to <input_dir>/build/.",
    )
    ap.add_argument(
        "--no-mmdc", action="store_true",
        help="Generate .md files but skip SVG rendering via mmdc.",
    )
    ap.add_argument(
        "--no-diagram", action="store_true",
        help="Skip diagram generation entirely (implies --no-mmdc).",
    )
    ap.add_argument(
        "--quiet", action="store_true",
        help="Suppress the AST summary; only show lint output.",
    )
    args = ap.parse_args()

    input_path = Path(args.path)
    if not input_path.exists():
        print(_c(f"Error: path not found: {args.path}", RED), file=sys.stderr)
        sys.exit(1)

    # Collect source files
    if input_path.is_file():
        source_files = [input_path]
        default_out = input_path.parent
    elif input_path.is_dir():
        source_files = sorted(input_path.rglob("*.stc"))
        default_out = input_path / "build"
        if not source_files:
            print(_c(f"No .stc files found in {input_path}", YELLOW))
            sys.exit(0)
        print(_c(f"Found {len(source_files)} .stc file(s) in {input_path}", BOLD))
    else:
        print(_c(f"Error: {args.path} is neither a file nor a directory.", RED), file=sys.stderr)
        sys.exit(1)

    out_dir: Optional[Path] = Path(args.out) if args.out else (
        None if input_path.is_file() else default_out
    )

    # Compile each file
    any_error = False
    generated_mds: List[Path] = []

    for src in source_files:
        success, md_path = compile_one(
            source_path=src,
            out_dir=out_dir,
            quiet=args.quiet,
            no_diagram=args.no_diagram,
        )
        if not success:
            any_error = True
        if md_path:
            generated_mds.append(md_path)

    # Run mmdc on all generated .md files
    if generated_mds and not args.no_mmdc and not args.no_diagram:
        print(_c(f"\n⚙  Rendering {len(generated_mds)} diagram(s) with mmdc …", BOLD))
        mmdc_ok = True
        for md in generated_mds:
            if not render_svg(md):
                mmdc_ok = False
                any_error = True
        if mmdc_ok:
            print(_c(f"   ✓  All SVGs rendered.", GREEN))

    # Summary
    n = len(source_files)
    n_ok = n - sum(1 for _ in range(n) if any_error)  # rough
    print()
    if any_error:
        print(_c(f"Finished with errors. ({len(generated_mds)}/{n} diagrams generated)", RED))
        sys.exit(1)
    else:
        print(_c(
            f"✓  Done. {n} file(s) compiled"
            + (f", {len(generated_mds)} diagram(s) written" if generated_mds else "")
            + ("." if args.no_mmdc or args.no_diagram else f", {len(generated_mds)} SVG(s) rendered."),
            GREEN,
        ))


if __name__ == "__main__":
    main()
