"""
STC Parser
==========
Hand-written recursive-descent parser for .stc source files.

Produces a ServiceAST from source text.

Grammar (informal):
    file        := service_decl external* routine* state*
    service     := 'service' IDENT
    external    := 'external' ('DB'|'API'|'DATA') IDENT
    routine     := 'routine' IDENT '(' dep_list ')' '{' routine_body '}'
    dep_list    := IDENT (',' IDENT)*  |  ε
    routine_body:= ( do_stmt | rollback_stmt | lock_stmt )*
    do_stmt     := 'do' STRING ( 'to' IDENT )?
    rollback_stmt:= 'rollback' STRING ( 'to' IDENT )?
    lock_stmt   := 'lock' IDENT
    state       := ('first'|'terminal')* 'state' IDENT '{' state_body '}'
    state_body  := ( has_stmt | steps_stmt | transition )*
    has_stmt    := 'has' '[' ident_list ']'
    steps_stmt  := 'steps' '[' ident_list ']'
    transition  := 'on' IDENT '->' IDENT
    ident_list  := IDENT (',' IDENT)*  |  ε
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from stc_ast import (
    ExternalType,
    External,
    RoutineAction,
    Routine,
    Transition,
    State,
    ServiceAST,
)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    def __init__(self, message: str, line: int):
        super().__init__(f"Line {line}: {message}")
        self.line = line


# Token types
TK_IDENT   = "IDENT"
TK_STRING  = "STRING"
TK_ARROW   = "ARROW"     # ->
TK_LBRACE  = "LBRACE"    # {
TK_RBRACE  = "RBRACE"    # }
TK_LBRACKET = "LBRACKET" # [
TK_RBRACKET = "RBRACKET" # ]
TK_LPAREN  = "LPAREN"    # (
TK_RPAREN  = "RPAREN"    # )
TK_COMMA   = "COMMA"     # ,
TK_EOF     = "EOF"


Token = Tuple[str, str, int]  # (type, value, line_number)


def _strip_comments(source: str) -> str:
    """Remove // and # line comments, preserving line numbers."""
    lines = source.splitlines()
    cleaned = []
    for line in lines:
        # Remove // comments (but not inside strings — good enough for this language)
        line = re.sub(r'//.*', '', line)
        # Remove # comments (but not inside strings)
        line = re.sub(r'#.*', '', line)
        cleaned.append(line)
    return '\n'.join(cleaned)


def tokenize(source: str) -> List[Token]:
    source = _strip_comments(source)
    tokens: List[Token] = []
    pos = 0
    line = 1

    while pos < len(source):
        # skip whitespace
        if source[pos] in ' \t\r':
            pos += 1
            continue
        if source[pos] == '\n':
            line += 1
            pos += 1
            continue

        # Arrow ->
        if source[pos:pos+2] == '->':
            tokens.append((TK_ARROW, '->', line))
            pos += 2
            continue

        # Single-char tokens
        ch = source[pos]
        if ch == '{':
            tokens.append((TK_LBRACE, '{', line))
            pos += 1
            continue
        if ch == '}':
            tokens.append((TK_RBRACE, '}', line))
            pos += 1
            continue
        if ch == '[':
            tokens.append((TK_LBRACKET, '[', line))
            pos += 1
            continue
        if ch == ']':
            tokens.append((TK_RBRACKET, ']', line))
            pos += 1
            continue
        if ch == '(':
            tokens.append((TK_LPAREN, '(', line))
            pos += 1
            continue
        if ch == ')':
            tokens.append((TK_RPAREN, ')', line))
            pos += 1
            continue
        if ch == ',':
            tokens.append((TK_COMMA, ',', line))
            pos += 1
            continue

        # String literal (double-quoted)
        if ch == '"':
            end = pos + 1
            while end < len(source) and source[end] != '"':
                if source[end] == '\n':
                    raise ParseError("Unterminated string literal", line)
                end += 1
            if end >= len(source):
                raise ParseError("Unterminated string literal", line)
            tokens.append((TK_STRING, source[pos+1:end], line))
            pos = end + 1
            continue

        # Identifier or keyword
        m = re.match(r'[A-Za-z_][A-Za-z0-9_]*', source[pos:])
        if m:
            tokens.append((TK_IDENT, m.group(), line))
            pos += m.end()
            continue

        raise ParseError(f"Unexpected character: {ch!r}", line)

    tokens.append((TK_EOF, '', line))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._pos = 0

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def _current(self) -> Token:
        return self._tokens[self._pos]

    @property
    def _line(self) -> int:
        return self._current[2]

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[idx]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if self._pos < len(self._tokens) - 1:
            self._pos += 1
        return tok

    def _expect(self, type_: str, value: Optional[str] = None) -> Token:
        tok = self._current
        if tok[0] != type_:
            raise ParseError(
                f"Expected {type_!r} but got {tok[0]!r} ({tok[1]!r})", tok[2]
            )
        if value is not None and tok[1] != value:
            raise ParseError(
                f"Expected {value!r} but got {tok[1]!r}", tok[2]
            )
        return self._advance()

    def _match(self, type_: str, value: Optional[str] = None) -> bool:
        tok = self._current
        if tok[0] != type_:
            return False
        if value is not None and tok[1] != value:
            return False
        return True

    def _consume_if(self, type_: str, value: Optional[str] = None) -> Optional[Token]:
        if self._match(type_, value):
            return self._advance()
        return None

    # -----------------------------------------------------------------------
    # Top-level
    # -----------------------------------------------------------------------

    def parse(self) -> ServiceAST:
        service_name = self._parse_service()
        externals: List[External] = []
        routines: List[Routine] = []
        states: List[State] = []

        while not self._match(TK_EOF):
            if self._match(TK_IDENT, 'external'):
                externals.append(self._parse_external())
            elif self._match(TK_IDENT, 'routine'):
                routines.append(self._parse_routine(is_locked=False))
            elif self._match(TK_IDENT, 'first'):
                states.append(self._parse_state(is_first=True, is_terminal=False))
            elif self._match(TK_IDENT, 'terminal'):
                states.append(self._parse_state(is_first=False, is_terminal=True))
            elif self._match(TK_IDENT, 'state'):
                states.append(self._parse_state(is_first=False, is_terminal=False))
            else:
                tok = self._current
                raise ParseError(
                    f"Unexpected token {tok[1]!r} at top level", tok[2]
                )

        return ServiceAST(
            service_name=service_name,
            externals=externals,
            routines=routines,
            states=states,
        )

    # -----------------------------------------------------------------------
    # service
    # -----------------------------------------------------------------------

    def _parse_service(self) -> str:
        self._expect(TK_IDENT, 'service')
        name_tok = self._expect(TK_IDENT)
        return name_tok[1]

    # -----------------------------------------------------------------------
    # external
    # -----------------------------------------------------------------------

    def _parse_external(self) -> External:
        line = self._line
        self._expect(TK_IDENT, 'external')
        type_tok = self._expect(TK_IDENT)
        raw_type = type_tok[1].upper()
        try:
            ext_type = ExternalType(raw_type)
        except ValueError:
            raise ParseError(
                f"Unknown external type {type_tok[1]!r}. Expected DB, API or DATA",
                type_tok[2],
            )
        name_tok = self._expect(TK_IDENT)
        return External(name=name_tok[1], type=ext_type, line=line)

    # -----------------------------------------------------------------------
    # routine
    # -----------------------------------------------------------------------

    def _parse_routine(self, is_locked: bool = False) -> Routine:
        line = self._line
        self._expect(TK_IDENT, 'routine')
        name_tok = self._expect(TK_IDENT)
        name = name_tok[1]

        self._expect(TK_LPAREN)
        deps = self._parse_ident_list_inner(TK_RPAREN)
        self._expect(TK_RPAREN)

        self._expect(TK_LBRACE)
        action: Optional[RoutineAction] = None
        rollback: Optional[RoutineAction] = None
        lock_target: Optional[str] = None

        while not self._match(TK_RBRACE):
            if self._match(TK_EOF):
                raise ParseError(f"Unclosed routine '{name}'", line)

            if self._match(TK_IDENT, 'do'):
                if action is not None:
                    raise ParseError(f"Duplicate 'do' in routine '{name}'", self._line)
                action = self._parse_action_stmt(is_rollback=False)
            elif self._match(TK_IDENT, 'rollback'):
                if rollback is not None:
                    raise ParseError(f"Duplicate 'rollback' in routine '{name}'", self._line)
                rollback = self._parse_action_stmt(is_rollback=True)
            elif self._match(TK_IDENT, 'lock'):
                stmt_line = self._line
                self._advance()
                target_tok = self._expect(TK_IDENT)
                if lock_target is not None:
                    raise ParseError(f"Duplicate 'lock' in routine '{name}'", stmt_line)
                lock_target = target_tok[1]
            else:
                tok = self._current
                raise ParseError(
                    f"Unexpected token {tok[1]!r} inside routine '{name}'", tok[2]
                )

        self._expect(TK_RBRACE)

        if action is None:
            raise ParseError(f"Routine '{name}' has no 'do' statement", line)

        return Routine(
            name=name,
            dependencies=deps,
            action=action,
            rollback=rollback,
            lock_target=lock_target,
            line=line,
        )

    def _parse_action_stmt(self, is_rollback: bool) -> RoutineAction:
        line = self._line
        keyword = 'rollback' if is_rollback else 'do'
        self._expect(TK_IDENT, keyword)
        query_tok = self._expect(TK_STRING)
        target: Optional[str] = None
        if self._consume_if(TK_IDENT, 'to'):
            target_tok = self._expect(TK_IDENT)
            target = target_tok[1]
        return RoutineAction(query=query_tok[1], target=target, is_rollback=is_rollback, line=line)

    # -----------------------------------------------------------------------
    # state
    # -----------------------------------------------------------------------

    def _parse_state(self, is_first: bool, is_terminal: bool) -> State:
        line = self._line

        # Consume leading keyword(s) before 'state'
        if is_first:
            self._expect(TK_IDENT, 'first')
        if is_terminal:
            self._expect(TK_IDENT, 'terminal')
        self._expect(TK_IDENT, 'state')

        name_tok = self._expect(TK_IDENT)
        name = name_tok[1]

        self._expect(TK_LBRACE)

        has_deps: List[str] = []
        steps: List[str] = []
        transitions: List[Transition] = []

        while not self._match(TK_RBRACE):
            if self._match(TK_EOF):
                raise ParseError(f"Unclosed state '{name}'", line)

            if self._match(TK_IDENT, 'has'):
                if has_deps:
                    raise ParseError(f"Duplicate 'has' in state '{name}'", self._line)
                has_deps = self._parse_bracketed_ident_list()
            elif self._match(TK_IDENT, 'steps'):
                if steps:
                    raise ParseError(f"Duplicate 'steps' in state '{name}'", self._line)
                steps = self._parse_bracketed_ident_list()
            elif self._match(TK_IDENT, 'on'):
                transitions.append(self._parse_transition(state_name=name))
            else:
                tok = self._current
                raise ParseError(
                    f"Unexpected token {tok[1]!r} inside state '{name}'", tok[2]
                )

        self._expect(TK_RBRACE)

        return State(
            name=name,
            is_first=is_first,
            is_terminal=is_terminal,
            has_deps=has_deps,
            steps=steps,
            transitions=transitions,
            line=line,
        )

    def _parse_transition(self, state_name: str) -> Transition:
        line = self._line
        self._expect(TK_IDENT, 'on')
        event_tok = self._expect(TK_IDENT)
        self._expect(TK_ARROW)
        target_tok = self._expect(TK_IDENT)
        return Transition(event=event_tok[1], target_state=target_tok[1], line=line)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _parse_bracketed_ident_list(self) -> List[str]:
        """Parse `keyword [ ident, ident, ... ]` — consumes the keyword too."""
        self._advance()  # consume keyword ('has' or 'steps')
        self._expect(TK_LBRACKET)
        items = self._parse_ident_list_inner(TK_RBRACKET)
        self._expect(TK_RBRACKET)
        return items

    def _parse_ident_list_inner(self, stop_type: str) -> List[str]:
        """Parse comma-separated identifiers until stop token (not consumed)."""
        items: List[str] = []
        while not self._match(stop_type) and not self._match(TK_EOF):
            tok = self._expect(TK_IDENT)
            items.append(tok[1])
            if not self._consume_if(TK_COMMA):
                break
        return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(source: str) -> ServiceAST:
    """Parse a .stc source string and return a ServiceAST."""
    tokens = tokenize(source)
    return Parser(tokens).parse()


def parse_file(path: str) -> ServiceAST:
    """Read a .stc file and return a ServiceAST."""
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()
    return parse(source)
