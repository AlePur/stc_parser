"""
STC Linter
==========
Traverses a ServiceAST and enforces safety and consistency rules.

Produces two lists:
  - errors:   Must be fixed before Mermaid output is generated.
  - warnings: Should be reviewed, but don't block output.

Rules
-----
ERRORS
  E001  Transition targets an undefined state.
  E002  Terminal state has outgoing transitions.
  E003  steps[] references an undefined routine.
  E004  Routine references an undefined external in its dependency list.
  E005  Exactly one `first` state must be declared.
  E006  `do/rollback ... to X` — target X is not in the routine's dependency list.
  E007  `do/rollback ... to X` — X is not a declared external.
  E008  `do lock X` — X (the resource) is not in the routine's dependency list.
  E009  `do lock X` — X is not a declared external.

WARNINGS
  W001  State has no outgoing transitions and is not marked terminal.
  W002  State is unreachable from the first state.
  W003  External declared but never referenced in any routine.
  W004  Routine declared but never used in any state's steps.
  W005  Routine used in a state that has a `fail` transition, but some step
        has no rollback — failure may leave the system in a dirty state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set

from stc_ast import DoKind, DoAction, ServiceAST, State


# ---------------------------------------------------------------------------
# Diagnostic dataclass
# ---------------------------------------------------------------------------

@dataclass
class Diagnostic:
    code: str        # e.g. "E001"
    message: str
    line: int        # source line (0 = unknown / not applicable)

    def __str__(self) -> str:
        prefix = "ERROR  " if self.code.startswith("E") else "WARNING"
        loc = f" (line {self.line})" if self.line else ""
        return f"[{self.code}] {prefix}{loc}: {self.message}"


# ---------------------------------------------------------------------------
# Linter
# ---------------------------------------------------------------------------

class STCLinter:
    def __init__(self, ast: ServiceAST):
        self.ast = ast
        self.errors: List[Diagnostic] = []
        self.warnings: List[Diagnostic] = []

        self._state_map: Dict[str, State] = {s.name: s for s in ast.states}
        self._external_names: Set[str] = {e.name for e in ast.externals}
        self._routine_names: Set[str] = {r.name for r in ast.routines}

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def lint(self) -> tuple[List[Diagnostic], List[Diagnostic]]:
        """Run all checks. Returns (errors, warnings)."""
        self._check_single_first_state()     # E005
        self._check_undefined_targets()      # E001
        self._check_terminal_violations()    # E002
        self._check_undefined_routines()     # E003
        self._check_undefined_externals()    # E004
        self._check_action_targets()         # E006, E007
        self._check_lock_resources()         # E008, E009

        self._check_implicit_terminal()      # W001
        self._check_unreachable_states()     # W002
        self._check_unused_externals()       # W003
        self._check_unused_routines()        # W004
        self._check_missing_rollbacks()      # W005

        return self.errors, self.warnings

    # -----------------------------------------------------------------------
    # Error checks
    # -----------------------------------------------------------------------

    def _check_single_first_state(self):
        """E005 — exactly one `first` state must be declared."""
        first_states = [s for s in self.ast.states if s.is_first]
        if len(first_states) == 0:
            self.errors.append(Diagnostic(
                "E005", "No `first` state declared. The flow has no entry point.", 0
            ))
        elif len(first_states) > 1:
            for s in first_states[1:]:
                self.errors.append(Diagnostic(
                    "E005",
                    f"Duplicate `first` state '{s.name}'. Only one entry point is allowed.",
                    s.line,
                ))

    def _check_undefined_targets(self):
        """E001 — transition points to a state that doesn't exist."""
        for state in self.ast.states:
            for trans in state.transitions:
                if trans.target_state not in self._state_map:
                    self.errors.append(Diagnostic(
                        "E001",
                        f"State '{state.name}': transition `on {trans.event}` targets "
                        f"undefined state '{trans.target_state}'.",
                        trans.line,
                    ))

    def _check_terminal_violations(self):
        """E002 — terminal state must not have outgoing transitions."""
        for state in self.ast.states:
            if state.is_terminal and state.transitions:
                events = ", ".join(f"'{t.event}'" for t in state.transitions)
                self.errors.append(Diagnostic(
                    "E002",
                    f"Terminal state '{state.name}' has outgoing transitions: {events}.",
                    state.line,
                ))

    def _check_undefined_routines(self):
        """E003 — steps references a routine that isn't declared."""
        for state in self.ast.states:
            for step in state.steps:
                if step not in self._routine_names:
                    self.errors.append(Diagnostic(
                        "E003",
                        f"State '{state.name}': steps references undefined routine '{step}'.",
                        state.line,
                    ))

    def _check_undefined_externals(self):
        """E004 — routine dependency references an external that isn't declared."""
        for routine in self.ast.routines:
            for dep in routine.dependencies:
                if dep not in self._external_names:
                    self.errors.append(Diagnostic(
                        "E004",
                        f"Routine '{routine.name}': dependency '{dep}' is not a declared external.",
                        routine.line,
                    ))

    def _check_action_targets(self):
        """
        E006 — `do/rollback ... to X`: X not in the routine's dependency list.
        E007 — `do/rollback ... to X`: X is not a declared external.
        """
        for routine in self.ast.routines:
            all_actions: List[DoAction] = []
            for step in routine.steps:
                all_actions.append(step.do_action)
                if step.rollback_action:
                    all_actions.append(step.rollback_action)

            for action in all_actions:
                if action.target is None:
                    continue
                tgt = action.target
                kind_str = action.kind.value
                if tgt not in routine.dependencies:
                    self.errors.append(Diagnostic(
                        "E006",
                        f"Routine '{routine.name}': `{kind_str} ... to {tgt}` — "
                        f"'{tgt}' is not in the routine's dependency list {routine.dependencies}.",
                        action.line,
                    ))
                elif tgt not in self._external_names:
                    self.errors.append(Diagnostic(
                        "E007",
                        f"Routine '{routine.name}': `{kind_str} ... to {tgt}` — "
                        f"'{tgt}' is not a declared external.",
                        action.line,
                    ))

    def _check_lock_resources(self):
        """
        E008 — `do lock X`: resource X not in the routine's dependency list.
        E009 — `do lock X`: resource X is not a declared external.
        """
        for routine in self.ast.routines:
            for step in routine.steps:
                for action in [step.do_action] + ([step.rollback_action] if step.rollback_action else []):
                    if action.kind != DoKind.LOCK or action.resource is None:
                        continue
                    res = action.resource
                    if res not in routine.dependencies:
                        self.errors.append(Diagnostic(
                            "E008",
                            f"Routine '{routine.name}': `lock {res}` — "
                            f"'{res}' is not in the routine's dependency list {routine.dependencies}.",
                            action.line,
                        ))
                    elif res not in self._external_names:
                        self.errors.append(Diagnostic(
                            "E009",
                            f"Routine '{routine.name}': `lock {res}` — "
                            f"'{res}' is not a declared external.",
                            action.line,
                        ))

    # -----------------------------------------------------------------------
    # Warning checks
    # -----------------------------------------------------------------------

    def _check_implicit_terminal(self):
        """W001 — state with no transitions that isn't marked terminal."""
        for state in self.ast.states:
            if not state.is_terminal and not state.transitions:
                self.warnings.append(Diagnostic(
                    "W001",
                    f"State '{state.name}' has no outgoing transitions but is not marked `terminal`. "
                    f"Flow dies here.",
                    state.line,
                ))

    def _check_unreachable_states(self):
        """W002 — states not reachable from the first state via BFS."""
        first_states = [s for s in self.ast.states if s.is_first]
        if not first_states:
            return  # already caught by E005

        visited: Set[str] = set()
        queue = deque([first_states[0].name])

        while queue:
            current_name = queue.popleft()
            if current_name in visited:
                continue
            visited.add(current_name)
            state = self._state_map.get(current_name)
            if state is None:
                continue
            for trans in state.transitions:
                if trans.target_state not in visited:
                    queue.append(trans.target_state)

        for state in self.ast.states:
            if state.name not in visited:
                self.warnings.append(Diagnostic(
                    "W002",
                    f"State '{state.name}' is unreachable from the first state.",
                    state.line,
                ))

    def _check_unused_externals(self):
        """W003 — external declared but never referenced in any routine dependency."""
        referenced: Set[str] = set()
        for routine in self.ast.routines:
            referenced.update(routine.dependencies)

        for ext in self.ast.externals:
            if ext.name not in referenced:
                self.warnings.append(Diagnostic(
                    "W003",
                    f"External '{ext.name}' ({ext.type.value}) is declared but never used "
                    f"in any routine's dependency list.",
                    ext.line,
                ))

    def _check_unused_routines(self):
        """W004 — routine declared but never referenced in any state's steps."""
        used: Set[str] = set()
        for state in self.ast.states:
            used.update(state.steps)

        for routine in self.ast.routines:
            if routine.name not in used:
                self.warnings.append(Diagnostic(
                    "W004",
                    f"Routine '{routine.name}' is declared but never used in any state's steps.",
                    routine.line,
                ))

    def _check_missing_rollbacks(self):
        """
        W005 — a routine used in a state's steps has a step without a rollback,
        and that state has a `fail` transition.  Failure may leave system dirty.
        """
        routine_map = {r.name: r for r in self.ast.routines}

        for state in self.ast.states:
            has_fail = any(t.event == "fail" for t in state.transitions)
            if not has_fail:
                continue

            for step_name in state.steps:
                routine = routine_map.get(step_name)
                if routine is None:
                    continue  # already caught by E003
                if routine.is_unfailing:
                    continue  # unfailing routines never fail
                for i, step in enumerate(routine.steps):
                    if step.rollback_action is None:
                        self.warnings.append(Diagnostic(
                            "W005",
                            f"State '{state.name}': routine '{step_name}' step {i} has no "
                            f"`rollback` but the state has a `fail` transition. "
                            f"Failure may leave the system in a dirty state.",
                            state.line,
                        ))
                        break  # one warning per routine is enough
