"""
STC / STCT Parser
=================
Hand-written recursive-descent parser for .stc and .stct source files.

Both file types share the same tokeniser.  The `Parser` base class handles
.stc constructs; `TestParser` (a subclass) adds .stct-specific grammar.

Grammar (.stc, informal)
------------------------
    file        := service_decl external* routine* state*
    service     := 'service' IDENT
    external    := 'external' ('DB'|'API'|'DATA') IDENT
    routine     := 'routine' ['unfailing'] IDENT '(' dep_list ')' '{' routine_body '}'
    dep_list    := IDENT (',' IDENT)*  |  ε
    routine_body:= ( do_step )*
    do_step     := do_action rollback_action?
    do_action   := 'do' ( lock_act | unlock_act | query_act )
    rollback_action := 'rollback' ( lock_act | unlock_act | query_act )
    lock_act    := 'lock' IDENT ( 'to' IDENT )?
    unlock_act  := 'unlock' IDENT ( 'to' IDENT )?
    query_act   := STRING ( 'to' IDENT )?
    state       := ('first'|'terminal')* 'state' IDENT '{' state_body '}'
    state_body  := ( has_stmt | steps_stmt | transition )*
    has_stmt    := 'has' '{' ident_list '}'
    steps_stmt  := 'steps' '[' ident_list ']'
    transition  := 'on' IDENT '->' IDENT
    ident_list  := IDENT (',' IDENT)*  |  ε

Grammar (.stct additions)
--------------------------
    test_file   := service_import external* test*
    service_import := 'service' IDENT 'from' STRING
    test        := 'test' IDENT '{' test_body '}'
    test_body   := ( example_stmt | start_stmt | run_stmt | assert_stmt )*
    example_stmt:= 'example' IDENT IDENT
    start_stmt  := IDENT ':=' 'start' IDENT '(' arg_list ')'
    arg_list    := arg (',' arg)*  |  ε
    arg         := IDENT ':' IDENT  |  IDENT
    run_stmt    := 'run' ( run_target | '{' run_target* '}' )
    run_target  := IDENT 'to' IDENT ['^']
    assert_stmt := 'assert' IDENT IDENT '=' NUMBER
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from stc_ast import (
    DoKind, DoAction, RoutineStep,
    ExternalType, External, Routine, Transition, State, ServiceAST,
    ServiceImport, ExampleData, ArgBinding, StartInstance,
    RunTarget, RunCommand, Assertion, TestCase, TestFileAST,
)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    def __init__(self, message: str, line: int):
        super().__init__(f"Line {line}: {message}")
        self.line = line


# Token types — .stc + .stct
TK_IDENT    = "IDENT"
TK_STRING   = "STRING"
TK_NUMBER   = "NUMBER"
TK_ARROW    = "ARROW"      # ->
TK_ASSIGN   = "ASSIGN"     # :=
TK_COLON    = "COLON"      # :
TK_CARET    = "CARET"      # ^
TK_EQUALS   = "EQUALS"     # =
TK_LBRACE   = "LBRACE"    # {
TK_RBRACE   = "RBRACE"    # }
TK_LBRACKET = "LBRACKET"  # [
TK_RBRACKET = "RBRACKET"  # ]
TK_LPAREN   = "LPAREN"    # (
TK_RPAREN   = "RPAREN"    # )
TK_COMMA    = "COMMA"     # ,
TK_EOF      = "EOF"

Token = Tuple[str, str, int]   # (type, value, line_number)


def _strip_comments(source: str) -> str:
    """Remove // and # line comments, preserving line numbers."""
    lines = source.splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(r'//.*', '', line)
        line = re.sub(r'#.*', '', line)
        cleaned.append(line)
    return '\n'.join(cleaned)


def tokenize(source: str) -> List[Token]:
    source = _strip_comments(source)
    tokens: List[Token] = []
    pos = 0
    line = 1

    while pos < len(source):
        ch = source[pos]

        # whitespace
        if ch in ' \t\r':
            pos += 1
            continue
        if ch == '\n':
            line += 1
            pos += 1
            continue

        # two-char tokens (check before single-char)
        two = source[pos:pos+2]
        if two == '->':
            tokens.append((TK_ARROW, '->', line))
            pos += 2
            continue
        if two == ':=':
            tokens.append((TK_ASSIGN, ':=', line))
            pos += 2
            continue

        # single-char tokens
        if ch == '{':
            tokens.append((TK_LBRACE, '{', line)); pos += 1; continue
        if ch == '}':
            tokens.append((TK_RBRACE, '}', line)); pos += 1; continue
        if ch == '[':
            tokens.append((TK_LBRACKET, '[', line)); pos += 1; continue
        if ch == ']':
            tokens.append((TK_RBRACKET, ']', line)); pos += 1; continue
        if ch == '(':
            tokens.append((TK_LPAREN, '(', line)); pos += 1; continue
        if ch == ')':
            tokens.append((TK_RPAREN, ')', line)); pos += 1; continue
        if ch == ',':
            tokens.append((TK_COMMA, ',', line)); pos += 1; continue
        if ch == ':':
            tokens.append((TK_COLON, ':', line)); pos += 1; continue
        if ch == '^':
            tokens.append((TK_CARET, '^', line)); pos += 1; continue
        if ch == '=':
            tokens.append((TK_EQUALS, '=', line)); pos += 1; continue

        # string literal
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

        # number literal
        m = re.match(r'[0-9]+', source[pos:])
        if m:
            tokens.append((TK_NUMBER, m.group(), line))
            pos += m.end()
            continue

        # identifier / keyword
        m = re.match(r'[A-Za-z_][A-Za-z0-9_]*', source[pos:])
        if m:
            tokens.append((TK_IDENT, m.group(), line))
            pos += m.end()
            continue

        raise ParseError(f"Unexpected character: {ch!r}", line)

    tokens.append((TK_EOF, '', line))
    return tokens


# ---------------------------------------------------------------------------
# Base Parser  (.stc)
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
            raise ParseError(f"Expected {value!r} but got {tok[1]!r}", tok[2])
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
    # Top-level (.stc)
    # -----------------------------------------------------------------------

    def parse(self) -> ServiceAST:
        service_name = self._parse_service_name()
        externals: List[External] = []
        routines: List[Routine] = []
        states: List[State] = []

        while not self._match(TK_EOF):
            if self._match(TK_IDENT, 'external'):
                externals.append(self._parse_external())
            elif self._match(TK_IDENT, 'routine'):
                routines.append(self._parse_routine())
            elif self._match(TK_IDENT, 'first'):
                states.append(self._parse_state(is_first=True, is_terminal=False))
            elif self._match(TK_IDENT, 'terminal'):
                states.append(self._parse_state(is_first=False, is_terminal=True))
            elif self._match(TK_IDENT, 'state'):
                states.append(self._parse_state(is_first=False, is_terminal=False))
            else:
                tok = self._current
                raise ParseError(f"Unexpected token {tok[1]!r} at top level", tok[2])

        return ServiceAST(
            service_name=service_name,
            externals=externals,
            routines=routines,
            states=states,
        )

    # -----------------------------------------------------------------------
    # service
    # -----------------------------------------------------------------------

    def _parse_service_name(self) -> str:
        self._expect(TK_IDENT, 'service')
        return self._expect(TK_IDENT)[1]

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

    def _parse_routine(self) -> Routine:
        line = self._line
        self._expect(TK_IDENT, 'routine')

        is_unfailing = False
        if self._match(TK_IDENT, 'unfailing'):
            is_unfailing = True
            self._advance()

        name_tok = self._expect(TK_IDENT)
        name = name_tok[1]

        self._expect(TK_LPAREN)
        deps = self._parse_ident_list_inner(TK_RPAREN)
        self._expect(TK_RPAREN)

        self._expect(TK_LBRACE)
        steps: List[RoutineStep] = []
        pending_do: Optional[DoAction] = None

        while not self._match(TK_RBRACE):
            if self._match(TK_EOF):
                raise ParseError(f"Unclosed routine '{name}'", line)

            if self._match(TK_IDENT, 'do'):
                # Flush any pending do without a rollback
                if pending_do is not None:
                    steps.append(RoutineStep(do_action=pending_do, rollback_action=None))
                pending_do = self._parse_do_or_rollback(keyword='do')

            elif self._match(TK_IDENT, 'rollback'):
                rb = self._parse_do_or_rollback(keyword='rollback')
                if pending_do is None:
                    raise ParseError(
                        f"'rollback' without a preceding 'do' in routine '{name}'",
                        self._line,
                    )
                steps.append(RoutineStep(do_action=pending_do, rollback_action=rb))
                pending_do = None

            else:
                tok = self._current
                raise ParseError(
                    f"Unexpected token {tok[1]!r} inside routine '{name}'", tok[2]
                )

        # Flush trailing do
        if pending_do is not None:
            steps.append(RoutineStep(do_action=pending_do, rollback_action=None))

        self._expect(TK_RBRACE)

        if not steps:
            raise ParseError(f"Routine '{name}' has no steps (at least one 'do' required)", line)

        return Routine(
            name=name,
            is_unfailing=is_unfailing,
            dependencies=deps,
            steps=steps,
            line=line,
        )

    def _parse_do_or_rollback(self, keyword: str) -> DoAction:
        """Parse `do ...` or `rollback ...` — keyword already known to match."""
        line = self._line
        self._expect(TK_IDENT, keyword)

        if self._match(TK_IDENT, 'lock'):
            self._advance()
            resource = self._expect(TK_IDENT)[1]
            target: Optional[str] = None
            if self._consume_if(TK_IDENT, 'to'):
                target = self._expect(TK_IDENT)[1]
            return DoAction(kind=DoKind.LOCK, resource=resource, target=target, line=line)

        if self._match(TK_IDENT, 'unlock'):
            self._advance()
            resource = self._expect(TK_IDENT)[1]
            target = None
            if self._consume_if(TK_IDENT, 'to'):
                target = self._expect(TK_IDENT)[1]
            return DoAction(kind=DoKind.UNLOCK, resource=resource, target=target, line=line)

        # Query
        query = self._expect(TK_STRING)[1]
        target = None
        if self._consume_if(TK_IDENT, 'to'):
            target = self._expect(TK_IDENT)[1]
        return DoAction(kind=DoKind.QUERY, query=query, target=target, line=line)

    # -----------------------------------------------------------------------
    # state
    # -----------------------------------------------------------------------

    def _parse_state(self, is_first: bool, is_terminal: bool) -> State:
        line = self._line

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
                has_deps = self._parse_braced_ident_set()
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
    # Identifier list helpers
    # -----------------------------------------------------------------------

    def _parse_braced_ident_set(self) -> List[str]:
        """Parse `has { ident, ident, ... }` — consumes the leading keyword."""
        self._advance()            # consume 'has'
        self._expect(TK_LBRACE)
        items = self._parse_ident_list_inner(TK_RBRACE)
        self._expect(TK_RBRACE)
        return items

    def _parse_bracketed_ident_list(self) -> List[str]:
        """Parse `steps [ ident, ident, ... ]` — consumes the leading keyword."""
        self._advance()            # consume 'steps'
        self._expect(TK_LBRACKET)
        items = self._parse_ident_list_inner(TK_RBRACKET)
        self._expect(TK_RBRACKET)
        return items

    def _parse_ident_list_inner(self, stop_type: str) -> List[str]:
        """Parse comma-separated identifiers until `stop_type` (not consumed)."""
        items: List[str] = []
        while not self._match(stop_type) and not self._match(TK_EOF):
            tok = self._expect(TK_IDENT)
            items.append(tok[1])
            if not self._consume_if(TK_COMMA):
                break
        return items


# ---------------------------------------------------------------------------
# TestParser  (.stct)
# ---------------------------------------------------------------------------

class TestParser(Parser):
    """Parses .stct test definition files."""

    def parse_test(self) -> TestFileAST:
        service_import = self._parse_service_import()
        externals: List[External] = []
        tests: List[TestCase] = []

        while not self._match(TK_EOF):
            if self._match(TK_IDENT, 'external'):
                externals.append(self._parse_external())
            elif self._match(TK_IDENT, 'test'):
                tests.append(self._parse_test_case())
            else:
                tok = self._current
                raise ParseError(
                    f"Unexpected token {tok[1]!r} at top level of test file", tok[2]
                )

        return TestFileAST(
            service_import=service_import,
            externals=externals,
            tests=tests,
        )

    def _parse_service_import(self) -> ServiceImport:
        line = self._line
        self._expect(TK_IDENT, 'service')
        name = self._expect(TK_IDENT)[1]
        self._expect(TK_IDENT, 'from')
        path = self._expect(TK_STRING)[1]
        return ServiceImport(name=name, path=path, line=line)

    def _parse_test_case(self) -> TestCase:
        line = self._line
        self._expect(TK_IDENT, 'test')
        name = self._expect(TK_IDENT)[1]
        self._expect(TK_LBRACE)

        body = []
        while not self._match(TK_RBRACE):
            if self._match(TK_EOF):
                raise ParseError(f"Unclosed test '{name}'", line)

            if self._match(TK_IDENT, 'example'):
                body.append(self._parse_example())
            elif self._match(TK_IDENT) and self._peek(1)[0] == TK_ASSIGN:
                body.append(self._parse_start_instance())
            elif self._match(TK_IDENT, 'run'):
                body.append(self._parse_run_command())
            elif self._match(TK_IDENT, 'assert'):
                body.append(self._parse_assertion())
            else:
                tok = self._current
                raise ParseError(
                    f"Unexpected token {tok[1]!r} inside test '{name}'", tok[2]
                )

        self._expect(TK_RBRACE)
        return TestCase(name=name, body=body, line=line)

    def _parse_example(self) -> ExampleData:
        line = self._line
        self._expect(TK_IDENT, 'example')
        self._expect(TK_IDENT)   # type keyword (e.g. DATA) — parsed but not stored
        name = self._expect(TK_IDENT)[1]
        return ExampleData(name=name, line=line)

    def _parse_start_instance(self) -> StartInstance:
        line = self._line
        var_name = self._expect(TK_IDENT)[1]
        self._expect(TK_ASSIGN)
        self._expect(TK_IDENT, 'start')
        service_name = self._expect(TK_IDENT)[1]
        self._expect(TK_LPAREN)
        args = self._parse_arg_list()
        self._expect(TK_RPAREN)
        return StartInstance(var_name=var_name, service_name=service_name, args=args, line=line)

    def _parse_arg_list(self) -> List[ArgBinding]:
        """Parse `IDENT (',' IDENT)*`  or  `IDENT ':' IDENT` named bindings."""
        args: List[ArgBinding] = []
        while not self._match(TK_RPAREN) and not self._match(TK_EOF):
            name_tok = self._expect(TK_IDENT)
            if self._consume_if(TK_COLON):
                # Named binding: Key: Value
                value_tok = self._expect(TK_IDENT)
                args.append(ArgBinding(external_ref=value_tok[1], key=name_tok[1]))
            else:
                args.append(ArgBinding(external_ref=name_tok[1], key=None))
            if not self._consume_if(TK_COMMA):
                break
        return args

    def _parse_run_command(self) -> RunCommand:
        line = self._line
        self._expect(TK_IDENT, 'run')

        if self._match(TK_LBRACE):
            # Concurrent block: run { target1  target2  ... }
            self._advance()
            targets: List[RunTarget] = []
            while not self._match(TK_RBRACE) and not self._match(TK_EOF):
                targets.append(self._parse_run_target())
            self._expect(TK_RBRACE)
            return RunCommand(targets=targets, is_concurrent=True, line=line)

        # Single target
        target = self._parse_run_target()
        return RunCommand(targets=[target], is_concurrent=False, line=line)

    def _parse_run_target(self) -> RunTarget:
        line = self._line
        var_name = self._expect(TK_IDENT)[1]
        self._expect(TK_IDENT, 'to')
        state_tok = self._expect(TK_IDENT)
        state_name = state_tok[1]
        is_start = self._consume_if(TK_CARET) is not None
        return RunTarget(var_name=var_name, state_name=state_name, is_start=is_start, line=line)

    def _parse_assertion(self) -> Assertion:
        line = self._line
        self._expect(TK_IDENT, 'assert')
        ext_name = self._expect(TK_IDENT)[1]
        metric = self._expect(TK_IDENT)[1]    # e.g. "writes"
        self._expect(TK_EQUALS)
        value_tok = self._expect(TK_NUMBER)
        return Assertion(
            external_name=ext_name,
            metric=metric,
            operator="=",
            value=int(value_tok[1]),
            line=line,
        )


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


def parse_test(source: str) -> TestFileAST:
    """Parse a .stct source string and return a TestFileAST."""
    tokens = tokenize(source)
    return TestParser(tokens).parse_test()


def parse_test_file(path: str) -> TestFileAST:
    """Read a .stct file and return a TestFileAST."""
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()
    return parse_test(source)
