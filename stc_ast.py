"""
STC Abstract Syntax Tree (AST) Data Model
==========================================
Dataclasses representing every node in a parsed .stc or .stct file.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

class ExternalType(Enum):
    DB   = "DB"
    API  = "API"
    DATA = "DATA"


@dataclass
class External:
    """A declared external dependency: `external DB MainDB`"""
    name: str
    type: ExternalType
    line: int = 0


# ---------------------------------------------------------------------------
# Routine step model
# ---------------------------------------------------------------------------

class DoKind(Enum):
    QUERY  = "query"   # do "SQL..."  /  rollback "SQL..."
    LOCK   = "lock"    # do lock X to Y
    UNLOCK = "unlock"  # do unlock X  /  rollback unlock X to Y


@dataclass
class DoAction:
    """
    A single `do` or `rollback` action inside a routine step.

    Examples
    --------
    do "SELECT ..."              → kind=QUERY,  query="SELECT ...", resource=None, target=None
    do "SELECT ..." to MainDB   → kind=QUERY,  query="SELECT ...", resource=None, target="MainDB"
    do lock UserId to MainDB    → kind=LOCK,   query=None,         resource="UserId", target="MainDB"
    do unlock UserId            → kind=UNLOCK, query=None,         resource="UserId", target=None
    rollback unlock UserId to MainDB → kind=UNLOCK, resource="UserId", target="MainDB"
    """
    kind: DoKind
    query: Optional[str] = None       # for QUERY
    resource: Optional[str] = None    # for LOCK / UNLOCK
    target: Optional[str] = None      # external name the action is directed at
    line: int = 0


@dataclass
class RoutineStep:
    """
    A do / rollback pair within a routine body.

    Each `do` statement creates a new step.  The `rollback` that immediately
    follows it (before the next `do`) is paired with it.  If a `do` is not
    followed by a `rollback`, rollback_action is None.
    """
    do_action: DoAction
    rollback_action: Optional[DoAction] = None


@dataclass
class Routine:
    """
    A routine declaration.

    routine [unfailing] Name(dep, ...) {
        do ...
        rollback ...
        do ...
        rollback ...
    }

    `is_unfailing` — the routine is guaranteed never to fail; no failure
    branching is generated for its query steps during model checking.
    """
    name: str
    is_unfailing: bool
    dependencies: List[str]     # names of externals passed as args
    steps: List[RoutineStep]    # ordered list of do/rollback pairs
    line: int = 0


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """A state transition: `on <event> -> <target>`"""
    event: str
    target_state: str
    line: int = 0


@dataclass
class State:
    """
    A state node.

    [first] [terminal] state Name {
        has {Dep, ...}         # only on first state
        steps [Routine, ...]
        on event -> Target
    }
    """
    name: str
    is_first: bool = False
    is_terminal: bool = False
    has_deps: List[str] = field(default_factory=list)   # only on first state
    steps: List[str] = field(default_factory=list)       # routine names (ordered)
    transitions: List[Transition] = field(default_factory=list)
    line: int = 0


@dataclass
class ServiceAST:
    """Root node of a compiled .stc file."""
    service_name: str
    externals: List[External] = field(default_factory=list)
    routines: List[Routine] = field(default_factory=list)
    states: List[State] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Test file AST nodes  (.stct)
# ---------------------------------------------------------------------------

@dataclass
class ServiceImport:
    """service CheckoutService from "./checkout" """
    name: str
    path: str
    line: int = 0


@dataclass
class ExampleData:
    """example DATA ExampleUser — declares a fixture value inside a test."""
    name: str
    line: int = 0


@dataclass
class ArgBinding:
    """
    One argument in a `start(...)` call.

    Positional  — `ArgBinding(external_ref="MainDB")`
    Named       — `ArgBinding(external_ref="ExampleUser", key="UserId")`
                  means: the STC dep called 'UserId' is satisfied by the
                  test external / example named 'ExampleUser'.
    """
    external_ref: str
    key: Optional[str] = None   # STC dep name for named bindings


@dataclass
class StartInstance:
    """var := start ServiceName(args...)"""
    var_name: str
    service_name: str
    args: List[ArgBinding]
    line: int = 0


@dataclass
class RunTarget:
    """A single target within a run command: `var to State[^]`"""
    var_name: str
    state_name: str           # "termination"  or  a named state
    is_start: bool = False    # True if the `^` suffix is present
    line: int = 0


@dataclass
class RunCommand:
    """
    run var to State^
    run { var to State, var2 to State2 }
    """
    targets: List[RunTarget]
    is_concurrent: bool = False
    line: int = 0


@dataclass
class Assertion:
    """assert ExternalName writes = N"""
    external_name: str
    metric: str      # e.g. "writes"
    operator: str    # e.g. "="
    value: int
    line: int = 0


@dataclass
class TestCase:
    """test Name { ... }"""
    name: str
    body: List   # List[ExampleData | StartInstance | RunCommand | Assertion]
    line: int = 0


@dataclass
class TestFileAST:
    """Root node of a parsed .stct file."""
    service_import: ServiceImport
    externals: List[External] = field(default_factory=list)
    tests: List[TestCase] = field(default_factory=list)
