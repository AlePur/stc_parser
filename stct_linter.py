"""
STCT Linter
===========
Validates a TestFileAST against its imported ServiceAST, catching semantic
errors and suspicious patterns before the test runner executes anything.

Rules
-----
ERRORS
  E101  `start` arg references a name not declared in test scope
        (not an external or example data).
  E102  `start` arg's service-scope slot is not in the service's first-state
        `has` set (unknown dependency slot).
  E103  `start` call does not cover all required service slots
        (missing entries from the first-state `has` set).
  E104  `start` call provides duplicate bindings for the same service slot.
  E105  Variable redeclaration  (same var name used in two `var := start` stmts).
  E106  `run` target references an undefined variable.
  E107  `assert` references an external not declared in the test file scope.
  E108  `start` uses a service name that doesn't match the imported service.

WARNINGS
  W101  External declared in the test file but never referenced in any
        `start` arg's test-scope name.
  W102  Example data declared but never referenced in any `start` arg's
        test-scope name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from stc_ast import (
    ServiceAST, TestFileAST, TestCase,
    ExampleData, StartInstance, RunCommand, RunTarget, Assertion,
    ArgBinding,
)


# ---------------------------------------------------------------------------
# Re-use the same Diagnostic type from stc_linter for consistency
# ---------------------------------------------------------------------------

@dataclass
class Diagnostic:
    code: str        # e.g. "E101"
    message: str
    line: int        # 0 = unknown

    def __str__(self) -> str:
        prefix = "ERROR  " if self.code.startswith("E") else "WARNING"
        loc = f" (line {self.line})" if self.line else ""
        return f"[{self.code}] {prefix}{loc}: {self.message}"


# ---------------------------------------------------------------------------
# Linter
# ---------------------------------------------------------------------------

class STCTLinter:
    """
    Lints a TestFileAST cross-referenced against its ServiceAST.

    Parameters
    ----------
    test_ast    : the parsed .stct file
    service_ast : the parsed .stc service file that the test imports
    """

    def __init__(self, test_ast: TestFileAST, service_ast: ServiceAST) -> None:
        self.test_ast = test_ast
        self.service_ast = service_ast
        self.errors: List[Diagnostic] = []
        self.warnings: List[Diagnostic] = []

        # Build lookup tables for the service
        first_states = [s for s in service_ast.states if s.is_first]
        self._service_has: Set[str] = (
            set(first_states[0].has_deps) if first_states else set()
        )
        self._service_name = service_ast.service_name

        # Build test-scope name sets: externals and examples declared at file level
        # (These are per-file, shared across all tests.)
        self._test_externals: Dict[str, int] = {
            e.name: e.line for e in test_ast.externals
        }

    # -----------------------------------------------------------------------
    # Per-test checks
    # -----------------------------------------------------------------------

    def _lint_test(self, test: TestCase) -> None:
        # Build the test-local scope: file-level externals + per-test examples
        # Track which names are ever *used* (for W101/W102)
        test_examples: Dict[str, int] = {}   # name → line
        used_test_scope: Set[str] = set()

        # Track declared variables (service instances) in this test
        declared_vars: Dict[str, int] = {}   # var_name → line of first declaration

        # First pass: collect example declarations
        for stmt in test.body:
            if isinstance(stmt, ExampleData):
                test_examples[stmt.name] = stmt.line

        # ---- combined test scope (externals + examples) ----
        def _in_scope(name: str) -> bool:
            return name in self._test_externals or name in test_examples

        # Second pass: validate each statement
        for stmt in test.body:

            # ---- example: just record (already collected above) ----
            if isinstance(stmt, ExampleData):
                pass  # handled above

            # ---- start ----
            elif isinstance(stmt, StartInstance):
                # E108 — service name mismatch
                if stmt.service_name != self._service_name:
                    self.errors.append(Diagnostic(
                        "E108",
                        f"Test '{test.name}': `start {stmt.service_name}` — "
                        f"'{stmt.service_name}' does not match the imported service "
                        f"'{self._service_name}'.",
                        stmt.line,
                    ))

                # E105 — variable redeclaration
                if stmt.var_name in declared_vars:
                    self.errors.append(Diagnostic(
                        "E105",
                        f"Test '{test.name}': variable '{stmt.var_name}' is redeclared. "
                        f"First declared at line {declared_vars[stmt.var_name]}.",
                        stmt.line,
                    ))
                else:
                    declared_vars[stmt.var_name] = stmt.line

                # Validate args
                self._lint_start_args(
                    test_name=test.name,
                    stmt=stmt,
                    in_scope=_in_scope,
                    used_test_scope=used_test_scope,
                )

            # ---- run ----
            elif isinstance(stmt, RunCommand):
                for rt in stmt.targets:
                    # E106 — undefined variable
                    if rt.var_name not in declared_vars:
                        self.errors.append(Diagnostic(
                            "E106",
                            f"Test '{test.name}': `run` target '{rt.var_name}' is not "
                            f"a declared service instance variable.",
                            rt.line,
                        ))

            # ---- assert ----
            elif isinstance(stmt, Assertion):
                # E107 — external not in test scope
                if stmt.external_name not in self._test_externals:
                    self.errors.append(Diagnostic(
                        "E107",
                        f"Test '{test.name}': `assert {stmt.external_name} ...` — "
                        f"'{stmt.external_name}' is not declared as an external in the "
                        f"test file.",
                        stmt.line,
                    ))
                else:
                    used_test_scope.add(stmt.external_name)

        # ---- W101: unused file-level externals (across this test) ----
        for ext_name, ext_line in self._test_externals.items():
            if ext_name not in used_test_scope:
                # Only warn once per external per file (we'll deduplicate at file level)
                pass  # handled at file level in _check_unused_file_externals

        # ---- W102: unused examples in this test ----
        for ex_name, ex_line in test_examples.items():
            if ex_name not in used_test_scope:
                self.warnings.append(Diagnostic(
                    "W102",
                    f"Test '{test.name}': example data '{ex_name}' is declared but "
                    f"never used in any `start` argument.",
                    ex_line,
                ))

        # Store which file-level externals were used by this test so we can
        # emit W101 after all tests are processed.
        if not hasattr(self, '_used_externals'):
            self._used_externals: Set[str] = set()
        self._used_externals.update(
            n for n in used_test_scope if n in self._test_externals
        )

    def _lint_start_args(
        self,
        test_name: str,
        stmt: StartInstance,
        in_scope,
        used_test_scope: Set[str],
    ) -> None:
        """Validate the argument list of one `start` call."""
        args = stmt.args

        # Build mapping: service_slot → (test_scope_name, line)
        # service_slot = arg.key if named, else arg.external_ref
        slot_map: Dict[str, Tuple[str, int]] = {}
        seen_slots: Dict[str, int] = {}  # for E104 duplicate detection

        for arg in args:
            service_slot = arg.key if arg.key else arg.external_ref
            test_scope_name = arg.external_ref
            arg_line = stmt.line  # ArgBinding has no separate line; use stmt line

            # E101 — test-scope name not declared
            if not in_scope(test_scope_name):
                self.errors.append(Diagnostic(
                    "E101",
                    f"Test '{test_name}': `start` arg '{test_scope_name}' is not "
                    f"declared as an external or example data in this test.",
                    arg_line,
                ))
            else:
                used_test_scope.add(test_scope_name)

            # E102 — service slot not in has set
            if service_slot not in self._service_has:
                self.errors.append(Diagnostic(
                    "E102",
                    f"Test '{test_name}': `start` arg maps to service slot '{service_slot}' "
                    f"which is not in the service's required set {sorted(self._service_has)}.",
                    arg_line,
                ))
                continue

            # E104 — duplicate slot binding
            if service_slot in seen_slots:
                self.errors.append(Diagnostic(
                    "E104",
                    f"Test '{test_name}': service slot '{service_slot}' is bound more "
                    f"than once in the same `start` call.",
                    arg_line,
                ))
            else:
                seen_slots[service_slot] = arg_line
                slot_map[service_slot] = (test_scope_name, arg_line)

        # E103 — missing slots
        missing = self._service_has - set(seen_slots.keys())
        if missing:
            self.errors.append(Diagnostic(
                "E103",
                f"Test '{test_name}': `start {stmt.service_name}` is missing required "
                f"service argument(s): {sorted(missing)}.",
                stmt.line,
            ))

    # -----------------------------------------------------------------------
    # File-level warnings (called after all tests are processed)
    # -----------------------------------------------------------------------

    def _check_unused_file_externals(self) -> None:
        """W101 — file-level external declared but never used in any start arg."""
        used = getattr(self, '_used_externals', set())
        for ext in self.test_ast.externals:
            if ext.name not in used:
                self.warnings.append(Diagnostic(
                    "W101",
                    f"External '{ext.name}' ({ext.type.value}) is declared in the test "
                    f"file but never referenced in any `start` argument.",
                    ext.line,
                ))

    def lint(self) -> Tuple[List[Diagnostic], List[Diagnostic]]:
        """Run all checks. Returns (errors, warnings)."""
        for test in self.test_ast.tests:
            self._lint_test(test)
        self._check_unused_file_externals()
        return self.errors, self.warnings
