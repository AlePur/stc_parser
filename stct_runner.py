"""
STCT Runner
===========
Exhaustive model-checker for .stct test definition files.

Each test is run by exploring EVERY possible interleaving of concurrent
service instances and EVERY possible success/failure outcome for each
non-unfailing query step.  The set of all reachable terminal world states
is collected and checked against the test assertions.

Usage
-----
    python stct_runner.py example/checkout.stct
    python stct_runner.py tests/             # all .stct files in directory
    python stct_runner.py checkout.stct -v   # verbose (show every run-target)

Exit codes
----------
    0  All tests passed
    1  One or more tests failed or parse errors occurred
"""

from __future__ import annotations

import sys
import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from stc_ast import (
    DoKind, DoAction, RoutineStep,
    ServiceAST, State, Routine,
    TestFileAST, TestCase, StartInstance, RunCommand, RunTarget,
    ExampleData, Assertion,
)
from stc_parser import parse_file, parse_test_file, ParseError
from stct_linter import STCTLinter, Diagnostic


# ---------------------------------------------------------------------------
# ANSI colours
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


# ---------------------------------------------------------------------------
# World-state types
# ---------------------------------------------------------------------------

# An active lock entry: (resource_name, target_external_name)
# e.g. ("UserId", "MainDB")  from  do lock UserId to MainDB
Lock = Tuple[str, str]

# Phase constants for an instance
PHASE_EXEC  = "exec"   # about to execute do[step_idx] of routine[routine_idx]
PHASE_RB    = "rb"     # rolling back step[step_idx] of routine[routine_idx]
PHASE_TRANS = "trans"  # all steps done, take transition with `outcome`
PHASE_DONE  = "done"   # reached terminal state, everything complete


@dataclass(frozen=True)
class InstancePos:
    """
    Fully hashable position of one service instance inside the state machine.

    routine_idx  — which routine in state.steps[] we are currently executing
    step_idx     — which RoutineStep within that routine (0 = before first do)
    phase        — EXEC | RB | TRANS | DONE
    outcome      — "success" | "fail" | "" (only meaningful in PHASE_TRANS)
    """
    state_name: str
    phase: str
    routine_idx: int
    step_idx: int
    outcome: str


@dataclass(frozen=True)
class WorldState:
    """
    Fully hashable snapshot of ALL concurrent instances + shared external state.

    instances  — tuple of InstancePos, one per running instance (ordered by start)
    locks      — set of active (resource, target_external) lock entries
    writes     — write count per external, indexed by the simulator's external list
    """
    instances: Tuple[InstancePos, ...]
    locks: FrozenSet[Lock]
    writes: Tuple[int, ...]


# ---------------------------------------------------------------------------
# Service simulator
# ---------------------------------------------------------------------------

class ServiceSimulator:
    """
    Wraps a ServiceAST and provides single-step atomic advancement.

    Each call to `successors()` returns ALL possible next worlds from advancing
    one instance by exactly one atomic action (a single `do` or `rollback`
    statement, or a state transition).
    """

    def __init__(self, service_ast: ServiceAST) -> None:
        self.ast = service_ast
        self.external_names: List[str] = [e.name for e in service_ast.externals]
        self._ext_idx: Dict[str, int] = {n: i for i, n in enumerate(self.external_names)}
        self._state_map: Dict[str, State] = {s.name: s for s in service_ast.states}
        self._routine_map: Dict[str, Routine] = {r.name: r for r in service_ast.routines}

    # ------------------------------------------------------------------
    # Helpers — position construction
    # ------------------------------------------------------------------

    def initial_pos(self) -> InstancePos:
        """Position of a freshly started instance (first state, no steps run)."""
        first = next(s for s in self.ast.states if s.is_first)
        return self._enter_state(first.name)

    def _enter_state(self, state_name: str) -> InstancePos:
        """Build the InstancePos for an instance that just entered `state_name`."""
        state = self._state_map[state_name]
        if state.is_terminal and len(state.steps) == 0:
            return InstancePos(state_name, PHASE_DONE, 0, 0, "")
        if len(state.steps) == 0:
            # No routines → immediately ready to transition
            return InstancePos(state_name, PHASE_TRANS, 0, 0, "success")
        return InstancePos(state_name, PHASE_EXEC, 0, 0, "")

    # ------------------------------------------------------------------
    # Helpers — lookups
    # ------------------------------------------------------------------

    def _get_routine(self, pos: InstancePos) -> Optional[Routine]:
        state = self._state_map[pos.state_name]
        if pos.routine_idx >= len(state.steps):
            return None
        return self._routine_map.get(state.steps[pos.routine_idx])

    def _get_step(self, pos: InstancePos) -> Optional[RoutineStep]:
        routine = self._get_routine(pos)
        if routine is None:
            return None
        if pos.step_idx >= len(routine.steps):
            return None
        return routine.steps[pos.step_idx]

    def _resolve_target(self, action: DoAction, routine: Routine) -> Optional[str]:
        """
        Which external does this action write to?

        Explicit `to X` wins.  Otherwise, if the routine has exactly one
        dependency, use that as the shortcut target.
        """
        if action.target:
            return action.target
        if len(routine.dependencies) == 1:
            return routine.dependencies[0]
        return None

    def _add_write(self, writes: Tuple[int, ...], target: Optional[str]) -> Tuple[int, ...]:
        if target and target in self._ext_idx:
            w = list(writes)
            w[self._ext_idx[target]] += 1
            return tuple(w)
        return writes

    def _lock_target_for(self, resource: str, locks: FrozenSet[Lock]) -> Optional[str]:
        """Return the target external under which `resource` is locked, if any."""
        for res, tgt in locks:
            if res == resource:
                return tgt
        return None

    def _is_locked(self, resource: str, locks: FrozenSet[Lock]) -> bool:
        return any(res == resource for res, _ in locks)

    def _replace_instance(
        self,
        world: WorldState,
        inst_idx: int,
        new_pos: InstancePos,
        locks: Optional[FrozenSet[Lock]] = None,
        writes: Optional[Tuple[int, ...]] = None,
    ) -> WorldState:
        insts = list(world.instances)
        insts[inst_idx] = new_pos
        return WorldState(
            instances=tuple(insts),
            locks=locks if locks is not None else world.locks,
            writes=writes if writes is not None else world.writes,
        )

    # ------------------------------------------------------------------
    # Successor generation — one atomic step
    # ------------------------------------------------------------------

    def successors(
        self, world: WorldState, inst_idx: int
    ) -> List[Tuple[WorldState, str]]:
        """
        All possible next worlds produced by advancing instance `inst_idx`
        by exactly one atomic action.

        Returns list of (new_world, human_readable_description).
        """
        pos = world.instances[inst_idx]
        if pos.phase == PHASE_EXEC:
            return self._exec_step(world, inst_idx, pos)
        if pos.phase == PHASE_RB:
            return self._rollback_step(world, inst_idx, pos)
        if pos.phase == PHASE_TRANS:
            return self._take_transition(world, inst_idx, pos)
        return []   # PHASE_DONE — no moves

    # ---- EXEC phase ------------------------------------------------

    def _exec_step(
        self, world: WorldState, inst_idx: int, pos: InstancePos
    ) -> List[Tuple[WorldState, str]]:
        routine = self._get_routine(pos)
        step = self._get_step(pos)
        state = self._state_map[pos.state_name]

        if routine is None or step is None:
            # All routines exhausted in exec phase — move to next routine / transition
            return [self._advance_after_routine(world, inst_idx, pos, state, success=True)]

        do = step.do_action
        results: List[Tuple[WorldState, str]] = []

        # ---- LOCK ---------------------------------------------------
        if do.kind == DoKind.LOCK:
            resource = do.resource or ""   # parser always sets this for LOCK
            tgt = self._resolve_target(do, routine) or ""
            if self._is_locked(resource, world.locks):
                # Deterministic failure — lock already held
                desc = f"lock {resource} → CONFLICT (already locked)"
                results += self._begin_failure(world, inst_idx, pos, routine, desc)
            else:
                lock_entry: Lock = (resource, tgt)
                new_locks = world.locks | frozenset([lock_entry])
                new_writes = self._add_write(world.writes, tgt if tgt else None)
                new_pos = self._next_exec_pos(pos, routine, state)
                new_world = self._replace_instance(
                    world, inst_idx, new_pos, locks=new_locks, writes=new_writes
                )
                results.append((new_world, f"lock {resource} → {tgt or '(no target)'}"))

        # ---- UNLOCK -------------------------------------------------
        elif do.kind == DoKind.UNLOCK:
            resource = do.resource or ""   # parser always sets this for UNLOCK
            stored_tgt = self._lock_target_for(resource, world.locks)
            eff_tgt = do.target or stored_tgt  # explicit to overrides stored
            new_locks = frozenset((r, t) for r, t in world.locks if r != resource)
            new_writes = self._add_write(world.writes, eff_tgt)
            new_pos = self._next_exec_pos(pos, routine, state)
            new_world = self._replace_instance(
                world, inst_idx, new_pos, locks=new_locks, writes=new_writes
            )
            results.append((new_world, f"unlock {resource} → {eff_tgt or '(no target)'}"))

        # ---- QUERY --------------------------------------------------
        else:
            tgt = self._resolve_target(do, routine)
            if not routine.is_unfailing:
                # Nondeterministic: also explore the failure branch
                desc_fail = f"query '{do.query}' → FAIL"
                results += self._begin_failure(world, inst_idx, pos, routine, desc_fail)
            # Success branch (always generated)
            new_writes = self._add_write(world.writes, tgt)
            new_pos = self._next_exec_pos(pos, routine, state)
            new_world = self._replace_instance(world, inst_idx, new_pos, writes=new_writes)
            results.append((new_world, f"query '{do.query}' → OK, write {tgt}"))

        return results

    def _next_exec_pos(
        self, pos: InstancePos, routine: Routine, state: State
    ) -> InstancePos:
        """InstancePos after successfully completing do[step_idx]."""
        next_si = pos.step_idx + 1
        if next_si < len(routine.steps):
            return InstancePos(pos.state_name, PHASE_EXEC, pos.routine_idx, next_si, "")
        # Routine complete — advance to next routine or transition
        next_ri = pos.routine_idx + 1
        if next_ri < len(state.steps):
            return InstancePos(pos.state_name, PHASE_EXEC, next_ri, 0, "")
        # All routines done
        if state.is_terminal:
            return InstancePos(pos.state_name, PHASE_DONE, pos.routine_idx, next_si, "")
        return InstancePos(pos.state_name, PHASE_TRANS, pos.routine_idx, next_si, "success")

    def _advance_after_routine(
        self,
        world: WorldState,
        inst_idx: int,
        pos: InstancePos,
        state: State,
        success: bool,
    ) -> Tuple[WorldState, str]:
        """Called when no step remains in the current routine — advance."""
        next_ri = pos.routine_idx + 1
        if next_ri < len(state.steps):
            new_pos = InstancePos(pos.state_name, PHASE_EXEC, next_ri, 0, "")
            return self._replace_instance(world, inst_idx, new_pos), "advance to next routine"
        if state.is_terminal:
            new_pos = InstancePos(pos.state_name, PHASE_DONE, pos.routine_idx, pos.step_idx, "")
            return self._replace_instance(world, inst_idx, new_pos), "terminal → done"
        outcome = "success" if success else "fail"
        new_pos = InstancePos(pos.state_name, PHASE_TRANS, pos.routine_idx, pos.step_idx, outcome)
        return self._replace_instance(world, inst_idx, new_pos), f"all steps done → {outcome}"

    def _begin_failure(
        self,
        world: WorldState,
        inst_idx: int,
        pos: InstancePos,
        routine: Routine,
        desc: str,
    ) -> List[Tuple[WorldState, str]]:
        """
        Step pos.step_idx failed.
        Start rolling back step pos.step_idx-1, or go straight to TRANS fail
        if there is nothing to roll back.
        """
        if pos.step_idx == 0:
            new_pos = InstancePos(pos.state_name, PHASE_TRANS, pos.routine_idx, 0, "fail")
            return [(self._replace_instance(world, inst_idx, new_pos), desc + " → fail (no rollback)")]
        rb_idx = pos.step_idx - 1
        new_pos = InstancePos(pos.state_name, PHASE_RB, pos.routine_idx, rb_idx, "")
        return [(self._replace_instance(world, inst_idx, new_pos), desc + f" → rollback from step {rb_idx}")]

    # ---- ROLLBACK phase --------------------------------------------

    def _rollback_step(
        self, world: WorldState, inst_idx: int, pos: InstancePos
    ) -> List[Tuple[WorldState, str]]:
        """Execute rollback[step_idx], then continue rolling back or go to TRANS fail."""
        routine = self._get_routine(pos)
        step = routine.steps[pos.step_idx] if routine else None

        new_locks = world.locks
        new_writes = world.writes
        desc = f"rollback step {pos.step_idx}"

        if step and step.rollback_action and routine is not None:
            rb = step.rollback_action
            eff_tgt: Optional[str] = None

            if rb.kind == DoKind.UNLOCK:
                resource = rb.resource or ""
                stored_tgt = self._lock_target_for(resource, world.locks)
                eff_tgt = rb.target or stored_tgt
                new_locks = frozenset((r, t) for r, t in world.locks if r != resource)
                new_writes = self._add_write(world.writes, eff_tgt)
                desc = f"rollback: unlock {resource}"

            elif rb.kind == DoKind.LOCK:
                resource = rb.resource or ""
                eff_tgt = rb.target or ""
                lock_entry: Lock = (resource, eff_tgt)
                new_locks = world.locks | frozenset([lock_entry])
                new_writes = self._add_write(world.writes, eff_tgt if eff_tgt else None)
                desc = f"rollback: lock {resource}"

            else:  # QUERY
                eff_tgt = self._resolve_target(rb, routine)
                new_writes = self._add_write(world.writes, eff_tgt)
                desc = f"rollback: query '{rb.query}'"

        # Continue rolling back or finish
        if pos.step_idx > 0:
            new_pos = InstancePos(pos.state_name, PHASE_RB, pos.routine_idx, pos.step_idx - 1, "")
        else:
            new_pos = InstancePos(pos.state_name, PHASE_TRANS, pos.routine_idx, 0, "fail")

        new_world = self._replace_instance(
            world, inst_idx, new_pos, locks=new_locks, writes=new_writes
        )
        return [(new_world, desc)]

    # ---- TRANSITION phase ------------------------------------------

    def _take_transition(
        self, world: WorldState, inst_idx: int, pos: InstancePos
    ) -> List[Tuple[WorldState, str]]:
        state = self._state_map[pos.state_name]

        if state.is_terminal:
            new_pos = InstancePos(pos.state_name, PHASE_DONE, pos.routine_idx, pos.step_idx, "")
            return [(self._replace_instance(world, inst_idx, new_pos), "terminal → done")]

        matching = [t for t in state.transitions if t.event == pos.outcome]
        if not matching:
            # No transition for this outcome — dead end (stuck world)
            return []

        results = []
        for trans in matching:
            new_pos = self._enter_state(trans.target_state)
            new_world = self._replace_instance(world, inst_idx, new_pos)
            results.append((new_world, f"transition {pos.outcome} → {trans.target_state}"))
        return results

    # ------------------------------------------------------------------
    # Stop-condition check
    # ------------------------------------------------------------------

    def at_stop(self, pos: InstancePos, target: RunTarget) -> bool:
        """Return True when `pos` has reached the `target` position."""
        if target.state_name == "termination":
            return pos.phase == PHASE_DONE
        if target.is_start:
            # ^  means: just entered the state, no steps run yet
            return (
                pos.state_name == target.state_name
                and pos.phase == PHASE_EXEC
                and pos.routine_idx == 0
                and pos.step_idx == 0
            )
        # Without ^: state's steps complete (in trans or done phase)
        return pos.state_name == target.state_name and pos.phase in (PHASE_TRANS, PHASE_DONE)

    # ------------------------------------------------------------------
    # BFS exploration
    # ------------------------------------------------------------------

    def explore(
        self,
        initial_frontier: Set[WorldState],
        targets: Dict[int, RunTarget],
    ) -> Tuple[Set[WorldState], int]:
        """
        BFS from `initial_frontier`.

        Advances all instances in `targets` that have not yet reached their
        individual stop conditions.  Stops when every instance in every world
        has reached its target.

        Returns
        -------
        terminal_frontier : Set[WorldState]
            All distinct worlds in which every targeted instance is at its stop.
        states_explored : int
            Total number of unique WorldState nodes visited.
        """
        queue: deque[WorldState] = deque()
        visited: Set[WorldState] = set()

        for w in initial_frontier:
            if w not in visited:
                visited.add(w)
                queue.append(w)

        terminal: Set[WorldState] = set()

        while queue:
            world = queue.popleft()

            # All targeted instances at their stops?
            if all(self.at_stop(world.instances[i], t) for i, t in targets.items()):
                terminal.add(world)
                continue

            any_advance = False
            for inst_idx, tgt in targets.items():
                pos = world.instances[inst_idx]
                if self.at_stop(pos, tgt):
                    continue        # This instance is already done
                if pos.phase == PHASE_DONE:
                    continue        # No moves from DONE

                for new_world, _desc in self.successors(world, inst_idx):
                    if new_world not in visited:
                        visited.add(new_world)
                        queue.append(new_world)
                    any_advance = True

            if not any_advance:
                # Stuck world — no moves possible but targets not reached
                terminal.add(world)

        return terminal, len(visited)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestRunner:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def run_file(self, path: str) -> bool:
        try:
            test_ast = parse_test_file(path)
        except ParseError as e:
            print(_c(f"  Parse error in {path}: {e}", RED))
            return False
        except OSError as e:
            print(_c(f"  Cannot read {path}: {e}", RED))
            return False

        # Resolve service path relative to the .stct file
        base = Path(path).parent
        raw = test_ast.service_import.path
        if not raw.endswith(".stc"):
            raw += ".stc"
        service_path = (base / raw).resolve()

        try:
            service_ast = parse_file(str(service_path))
        except ParseError as e:
            print(_c(f"  Parse error in service {service_path}: {e}", RED))
            return False
        except OSError as e:
            print(_c(f"  Cannot read service {service_path}: {e}", RED))
            return False

        # ------------------------------------------------------------------
        # Lint the test file before running anything
        # ------------------------------------------------------------------
        linter = STCTLinter(test_ast, service_ast)
        errors, warnings = linter.lint()

        if warnings:
            print(_c(f"\n  ── Warnings ({len(warnings)}) ──────────────────────────────", DIM))
            for w in warnings:
                loc = _c(f"(line {w.line})", DIM) if w.line else ""
                code = _c(f"[{w.code}]", YELLOW, BOLD)
                print(f"  {code} {_c('WARNING', YELLOW)} {loc}")
                print(f"         {w.message}")

        if errors:
            print(_c(f"\n  ── Errors ({len(errors)}) ────────────────────────────────────", RED))
            for e in errors:
                loc = _c(f"(line {e.line})", DIM) if e.line else ""
                code = _c(f"[{e.code}]", RED, BOLD)
                print(f"  {code} {_c('ERROR  ', RED)} {loc}")
                print(f"         {e.message}")
            print(_c(
                f"\n  ✗  {len(errors)} compile error(s) — test execution aborted.\n",
                RED, BOLD,
            ))
            return False

        if not warnings:
            print(_c("  ✓  Compiled OK — no issues.", GREEN))
        else:
            print(_c(f"  ✓  Compiled with {len(warnings)} warning(s) — running tests.", YELLOW))

        # ------------------------------------------------------------------
        # Run tests
        # ------------------------------------------------------------------
        sim = ServiceSimulator(service_ast)
        all_passed = True

        for test in test_ast.tests:
            if not self._run_test(test, sim):
                all_passed = False

        return all_passed

    # ------------------------------------------------------------------

    def _run_test(self, test: TestCase, sim: ServiceSimulator) -> bool:
        print(_c(f"\n  test {test.name}", BOLD))

        n_ext = len(sim.external_names)
        initial_world = WorldState(
            instances=(),
            locks=frozenset(),
            writes=tuple(0 for _ in range(n_ext)),
        )
        frontier: Set[WorldState] = {initial_world}

        # var_name → instance index
        var_map: Dict[str, int] = {}
        passed = True

        for stmt in test.body:

            # ---- example: no simulation effect ----------------------
            if isinstance(stmt, ExampleData):
                if self.verbose:
                    print(f"    example {stmt.name}")

            # ---- start: add new instance to every world -------------
            elif isinstance(stmt, StartInstance):
                new_idx = len(next(iter(frontier)).instances)
                var_map[stmt.var_name] = new_idx
                init_pos = sim.initial_pos()
                frontier = {
                    WorldState(
                        instances=w.instances + (init_pos,),
                        locks=w.locks,
                        writes=w.writes,
                    )
                    for w in frontier
                }
                if self.verbose:
                    print(f"    start {stmt.var_name} → instance {new_idx}, "
                          f"frontier: {len(frontier)} world(s)")

            # ---- run: BFS until all targets reached -----------------
            elif isinstance(stmt, RunCommand):
                if not all(rt.var_name in var_map for rt in stmt.targets):
                    missing = [rt.var_name for rt in stmt.targets if rt.var_name not in var_map]
                    print(_c(f"    ✗  unknown variable(s): {missing}", RED))
                    passed = False
                    continue

                targets: Dict[int, RunTarget] = {
                    var_map[rt.var_name]: rt for rt in stmt.targets
                }
                target_desc = "  ".join(
                    f"{rt.var_name} → {rt.state_name}{'^ ' if rt.is_start else ''}"
                    for rt in stmt.targets
                )

                frontier, count = sim.explore(frontier, targets)
                print(
                    f"    run  {target_desc}"
                    f"  |  {_c(str(count), CYAN)} world states explored"
                    f"  |  {_c(str(len(frontier)), CYAN)} terminal world(s)"
                )

                if not frontier:
                    print(_c("    ✗  all execution paths are stuck (no valid transitions)", RED))
                    passed = False

            # ---- assert: check every terminal world -----------------
            elif isinstance(stmt, Assertion):
                ok, failures = self._check_assertion(frontier, stmt, sim)
                sym = _c("✓", GREEN, BOLD) if ok else _c("✗", RED, BOLD)
                status = _c("PASS", GREEN) if ok else _c("FAIL", RED)
                print(
                    f"    {sym}  assert {stmt.external_name} {stmt.metric} "
                    f"{stmt.operator} {stmt.value}  —  {status}"
                )
                if not ok:
                    passed = False
                    for w in list(failures)[:3]:
                        actual = w.writes[sim._ext_idx[stmt.external_name]]
                        lk = {r for r, _ in w.locks}
                        print(_c(
                            f"       counterexample: {stmt.external_name} {stmt.metric} = {actual}"
                            + (f",  open locks: {lk}" if lk else ""),
                            DIM,
                        ))

        # ---- implicit: no open locks at end of test -----------------
        lock_worlds = [w for w in frontier if w.locks]
        if lock_worlds:
            passed = False
            print(_c(f"    ✗  lock leak in {len(lock_worlds)} terminal world(s)", RED))
            for w in lock_worlds[:3]:
                print(_c(f"       open locks: {set(w.locks)}", DIM))
        else:
            print(_c("    ✓  no lock leaks", GREEN))

        verdict = _c("PASSED", GREEN, BOLD) if passed else _c("FAILED", RED, BOLD)
        print(f"  {test.name}: {verdict}")
        return passed

    # ------------------------------------------------------------------

    def _check_assertion(
        self,
        frontier: Set[WorldState],
        stmt: Assertion,
        sim: ServiceSimulator,
    ) -> Tuple[bool, Set[WorldState]]:
        """
        Check assertion universally (∀ worlds in frontier).
        Returns (all_pass, set_of_violating_worlds).
        """
        if stmt.external_name not in sim._ext_idx:
            # External not tracked → always fail
            return False, frontier

        idx = sim._ext_idx[stmt.external_name]
        failures: Set[WorldState] = set()

        for world in frontier:
            val = world.writes[idx]
            if stmt.operator == "=":
                ok = val == stmt.value
            else:
                ok = False  # unknown operator
            if not ok:
                failures.add(world)

        return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="stct_runner",
        description="Exhaustive model-checker for .stct test files.",
    )
    ap.add_argument("path", help="A .stct file or a directory containing .stct files.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print extra detail (start events, frontier sizes).")
    args = ap.parse_args()

    input_path = Path(args.path)
    if not input_path.exists():
        print(_c(f"Error: not found: {args.path}", RED), file=sys.stderr)
        sys.exit(1)

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob("*.stct"))
        if not files:
            print(_c(f"No .stct files found in {input_path}", YELLOW))
            sys.exit(0)

    runner = TestRunner(verbose=args.verbose)
    any_fail = False

    for f in files:
        print(_c(f"\n⚙  {f}", BOLD))
        if not runner.run_file(str(f)):
            any_fail = True

    print()
    if any_fail:
        print(_c("Finished with failures.", RED, BOLD))
        sys.exit(1)
    else:
        print(_c("All tests passed.", GREEN, BOLD))


if __name__ == "__main__":
    main()
