"""
STC Abstract Syntax Tree (AST) Data Model
==========================================
Dataclasses representing every node in a parsed .stc file.
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
    line: int = 0   # source line for error reporting


@dataclass
class RoutineAction:
    """
    A single `do` statement inside a routine body.
    `do "..." to Target`  — target is optional (shortcut for single-dep routines)
    """
    query: str
    target: Optional[str] = None   # name of the external dependency
    is_rollback: bool = False       # True if this is a `rollback` clause
    line: int = 0


@dataclass
class Routine:
    """
    A routine declaration.

    routine ReserveStock(MainDB) {
        do "UPDATE ..." to MainDB
        rollback "UPDATE ..." to MainDB
        lock SomeDep
    }
    """
    name: str
    dependencies: List[str]         # names of externals passed as args
    action: Optional[RoutineAction] = None
    rollback: Optional[RoutineAction] = None
    lock_target: Optional[str] = None  # name used in `lock <dep>`
    line: int = 0


@dataclass
class Transition:
    """
    A state transition: `on <event> -> <target>`
    """
    event: str
    target_state: str
    line: int = 0


@dataclass
class State:
    """
    A state node.

    [first] [terminal] state Name {
        has  [Dep, ...]       # only on first state
        steps [Routine, ...]
        on event -> Target
    }
    """
    name: str
    is_first: bool = False
    is_terminal: bool = False
    has_deps: List[str] = field(default_factory=list)   # only meaningful on first state
    steps: List[str] = field(default_factory=list)      # routine names
    transitions: List[Transition] = field(default_factory=list)
    line: int = 0


@dataclass
class ServiceAST:
    """Root node of a compiled .stc file."""
    service_name: str
    externals: List[External] = field(default_factory=list)
    routines: List[Routine] = field(default_factory=list)
    states: List[State] = field(default_factory=list)