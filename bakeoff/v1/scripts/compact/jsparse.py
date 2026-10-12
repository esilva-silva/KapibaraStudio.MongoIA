"""Tokenizer e parser (Pratt) de um subconjunto de JavaScript usado em mongosh e no Console da IDE.

Objetivos:
- Produzir AST com offsets exatos (start/end) para recortes FIM em fronteiras semânticas.
- Alimentar a validação estrutural MongoDB sem executar código.
Construções fora do subconjunto (classes, generators, labels...) geram ``Unsupported``: o exemplo
fica "não validável" em vez de inválido. A sintaxe JS completa é verificada pelo Node (syntax.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


class JsSyntaxError(ValueError):
    def __init__(self, message: str, pos: int) -> None:
        super().__init__(f"{message} (pos {pos})")
        self.pos = pos


class Unsupported(ValueError):
    pass


KEYWORDS = {
    "const", "let", "var", "function", "return", "if", "else", "for", "while", "do", "break", "continue",
    "new", "typeof", "instanceof", "in", "of", "delete", "void", "throw", "try", "catch", "finally",
    "class", "switch", "case", "default", "yield", "await", "async", "this", "null", "true", "false",
}
PUNCTUATORS = sorted([
    ">>>=", "...", "===", "!==", "**=", "<<=", ">>=", ">>>", "&&=", "||=", "??=",
    "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--", "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=", "**", "<<", ">>",
    "{", "}", "(", ")", "[", "]", ";", ",", "<", ">", "+", "-", "*", "/", "%", "&", "|", "^", "!", "~",
    "?", ":", "=", ".", "@", "#",
], key=len, reverse=True)
REGEX_PRECEDERS = {"(", ",", "=", ":", "[", "!", "&", "|", "?", "{", "}", ";", "+", "-", "*", "%", "<", ">",
                   "~", "^", "=>", "==", "===", "!=", "!==", "&&", "||", "??", "+=", "-=", "return", "typeof",
                   "case", "in", "of", "new", "delete", "void", "throw", "else", "do", "await", "yield"}


@dataclass(slots=True)
class Token:
    kind: str  # ident | num | str | template | regex | punct | eof
    value: str
    start: int
    end: int
    nl_before: bool = False


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(src)
    newline = False
    while i < n:
        c = src[i]
        if c in " \t\r﻿ ":
            i += 1
            continue
        if c == "\n":
            newline = True
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            if j < 0:
                raise JsSyntaxError("comentário não fechado", i)
            newline = newline or "\n" in src[i:j]
            i = j + 2
            continue
        start = i
        if c.isalpha() or c in "_$" or ord(c) > 127:
            while i < n and (src[i].isalnum() or src[i] in "_$" or ord(src[i]) > 127):
                i += 1
            tokens.append(Token("ident", src[start:i], start, i, newline))
        elif c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            if src.startswith(("0x", "0X", "0b", "0B", "0o", "0O"), i):
                i += 2
                while i < n and (src[i].isalnum() or src[i] == "_"):
                    i += 1
            else:
                while i < n and (src[i].isdigit() or src[i] in "._"):
                    i += 1
                if i < n and src[i] in "eE":
                    i += 1
                    if i < n and src[i] in "+-":
                        i += 1
                    while i < n and src[i].isdigit():
                        i += 1
                if i < n and src[i] == "n":
                    i += 1
            if i < n and (src[i].isalpha() or src[i] in "_$"):
                raise JsSyntaxError("número malformado", start)
            tokens.append(Token("num", src[start:i], start, i, newline))
        elif c in "'\"":
            i += 1
            while i < n and src[i] != c:
                if src[i] == "\\":
                    i += 1
                elif src[i] == "\n":
                    raise JsSyntaxError("string não fechada", start)
                i += 1
            if i >= n:
                raise JsSyntaxError("string não fechada", start)
            i += 1
            tokens.append(Token("str", src[start:i], start, i, newline))
        elif c == "`":
            i += 1
            depth = 0
            while i < n:
                ch = src[i]
                if ch == "\\":
                    i += 2
                    continue
                if depth == 0 and ch == "`":
                    break
                if src.startswith("${", i):
                    depth += 1
                    i += 2
                    continue
                if depth and ch == "}":
                    depth -= 1
                i += 1
            if i >= n:
                raise JsSyntaxError("template não fechado", start)
            i += 1
            tokens.append(Token("template", src[start:i], start, i, newline))
        elif c == "/" and _regex_allowed(tokens):
            i += 1
            in_class = False
            while i < n:
                ch = src[i]
                if ch == "\\":
                    i += 2
                    continue
                if ch == "\n":
                    raise JsSyntaxError("regex não fechada", start)
                if ch == "[":
                    in_class = True
                elif ch == "]":
                    in_class = False
                elif ch == "/" and not in_class:
                    break
                i += 1
            if i >= n:
                raise JsSyntaxError("regex não fechada", start)
            i += 1
            while i < n and src[i].isalpha():
                i += 1
            tokens.append(Token("regex", src[start:i], start, i, newline))
        else:
            for p in PUNCTUATORS:
                if src.startswith(p, i):
                    # "?." seguido de dígito é ternário + número
                    if p == "?." and i + 2 < n and src[i + 2].isdigit():
                        continue
                    i += len(p)
                    tokens.append(Token("punct", p, start, i, newline))
                    break
            else:
                raise JsSyntaxError(f"caractere inesperado {c!r}", i)
        newline = False
    tokens.append(Token("eof", "", n, n, newline))
    return tokens


def _regex_allowed(tokens: list[Token]) -> bool:
    if not tokens:
        return True
    prev = tokens[-1]
    if prev.kind == "punct":
        return prev.value in REGEX_PRECEDERS and prev.value not in (")", "]")
    if prev.kind == "ident":
        return prev.value in REGEX_PRECEDERS
    return False


class Node:
    __slots__ = ("type", "start", "end", "props")

    def __init__(self, type: str, start: int, end: int, **props: Any) -> None:
        self.type = type
        self.start = start
        self.end = end
        self.props = props

    def __getattr__(self, name: str) -> Any:
        try:
            return self.props[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, name: str, default: Any = None) -> Any:
        return self.props.get(name, default)

    def children(self) -> Iterator["Node"]:
        for value in self.props.values():
            if isinstance(value, Node):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Node):
                        yield item

    def __repr__(self) -> str:  # pragma: no cover - depuração
        return f"Node({self.type}, {self.start}:{self.end})"


def walk(node: Node) -> Iterator[Node]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(list(current.children())))


BINARY_PRECEDENCE = {
    "??": 1, "||": 2, "&&": 3, "|": 4, "^": 5, "&": 6,
    "==": 7, "!=": 7, "===": 7, "!==": 7,
    "<": 8, ">": 8, "<=": 8, ">=": 8, "instanceof": 8, "in": 8,
    "<<": 9, ">>": 9, ">>>": 9, "+": 10, "-": 10, "*": 11, "/": 11, "%": 11, "**": 12,
}
ASSIGNMENT = {"=", "+=", "-=", "*=", "/=", "%=", "**=", "<<=", ">>=", ">>>=", "&=", "|=", "^=", "&&=", "||=", "??="}


class Parser:
    def __init__(self, src: str) -> None:
        self.src = src
        self.tokens = tokenize(src)
        self.i = 0
        self.no_in = False

    # -- utilitários ------------------------------------------------------------------------------
    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def peek(self, offset: int = 1) -> Token:
        return self.tokens[min(self.i + offset, len(self.tokens) - 1)]

    def is_punct(self, value: str, token: Token | None = None) -> bool:
        token = token or self.tok
        return token.kind == "punct" and token.value == value

    def is_word(self, value: str, token: Token | None = None) -> bool:
        token = token or self.tok
        return token.kind == "ident" and token.value == value

    def advance(self) -> Token:
        token = self.tok
        if token.kind != "eof":
            self.i += 1
        return token

    def expect(self, value: str) -> Token:
        if not self.is_punct(value):
            raise JsSyntaxError(f"esperado {value!r}, encontrado {self.tok.value or self.tok.kind!r}", self.tok.start)
        return self.advance()

    def last_end(self) -> int:
        return self.tokens[self.i - 1].end if self.i else 0

    # -- programa e statements ----------------------------------------------------------------------
    def parse_program(self) -> Node:
        body = []
        while self.tok.kind != "eof":
            body.append(self.parse_statement())
        return Node("Program", 0, len(self.src), body=body)

    def consume_semicolon(self) -> None:
        if self.is_punct(";"):
            self.advance()
        elif not (self.tok.kind == "eof" or self.is_punct("}") or self.tok.nl_before):
            raise JsSyntaxError(f"';' ausente antes de {self.tok.value!r}", self.tok.start)

    def parse_statement(self) -> Node:
        tok = self.tok
        start = tok.start
        if self.is_punct(";"):
            self.advance()
            return Node("EmptyStatement", start, tok.end)
        if self.is_punct("{"):
            return self.parse_block()
        if tok.kind == "ident":
            word = tok.value
            if word in ("const", "let", "var") and not (word == "let" and self.peek().kind == "punct" and self.peek().value in ("=", ".", "(")):
                node = self.parse_variable_declaration()
                self.consume_semicolon()
                node.end = self.last_end()
                return node
            if word == "function" or (word == "async" and self.is_word("function", self.peek()) and not self.peek().nl_before):
                return self.parse_function(declaration=True)
            if word == "if":
                self.advance()
                self.expect("(")
                test = self.parse_expression()
                self.expect(")")
                consequent = self.parse_statement()
                alternate = None
                if self.is_word("else"):
                    self.advance()
                    alternate = self.parse_statement()
                return Node("IfStatement", start, self.last_end(), test=test, consequent=consequent, alternate=alternate)
            if word == "for":
                return self.parse_for()
            if word == "while":
                self.advance()
                self.expect("(")
                test = self.parse_expression()
                self.expect(")")
                body = self.parse_statement()
                return Node("WhileStatement", start, self.last_end(), test=test, body=body)
            if word in ("return", "throw"):
                self.advance()
                argument = None
                if not (self.is_punct(";") or self.is_punct("}") or self.tok.kind == "eof" or self.tok.nl_before):
                    argument = self.parse_expression()
                self.consume_semicolon()
                return Node("ReturnStatement" if word == "return" else "ThrowStatement", start, self.last_end(), argument=argument)
            if word in ("break", "continue"):
                self.advance()
                self.consume_semicolon()
                return Node("BreakStatement" if word == "break" else "ContinueStatement", start, self.last_end())
            if word == "try":
                self.advance()
                block = self.parse_block()
                handler = finalizer = None
                if self.is_word("catch"):
                    self.advance()
                    param = None
                    if self.is_punct("("):
                        self.advance()
                        param = self.parse_binding()
                        self.expect(")")
                    handler = Node("CatchClause", start, 0, param=param, body=self.parse_block())
                    handler.end = self.last_end()
                if self.is_word("finally"):
                    self.advance()
                    finalizer = self.parse_block()
                if handler is None and finalizer is None:
                    raise JsSyntaxError("try sem catch/finally", start)
                return Node("TryStatement", start, self.last_end(), block=block, handler=handler, finalizer=finalizer)
            if word in ("class", "switch", "do", "with", "import", "export", "yield"):
                raise Unsupported(word)
        expression = self.parse_expression()
        self.consume_semicolon()
        return Node("ExpressionStatement", start, self.last_end(), expression=expression)

    def parse_block(self) -> Node:
        start = self.expect("{").start
        body = []
        while not self.is_punct("}"):
            if self.tok.kind == "eof":
                raise JsSyntaxError("bloco não fechado", start)
            body.append(self.parse_statement())
        self.advance()
        return Node("BlockStatement", start, self.last_end(), body=body)

    def parse_binding(self) -> Node:
        tok = self.tok
        if tok.kind == "ident" and tok.value not in KEYWORDS - {"of", "async"}:
            self.advance()
            return Node("Identifier", tok.start, tok.end, name=tok.value)
        if self.is_punct("{") or self.is_punct("["):
            return self.parse_primary()  # padrão de destructuring tratado como literal
        raise JsSyntaxError("identificador esperado", tok.start)

    def parse_variable_declaration(self) -> Node:
        kind_tok = self.advance()
        declarations = []
        while True:
            target = self.parse_binding()
            init = None
            if self.is_punct("="):
                self.advance()
                init = self.parse_assignment()
            elif kind_tok.value == "const" and not (self.is_word("of") or self.is_word("in")):
                raise JsSyntaxError("const sem inicializador", target.start)
            declarations.append(Node("VariableDeclarator", target.start, self.last_end(), id=target, init=init))
            if not self.is_punct(","):
                break
            self.advance()
        return Node("VariableDeclaration", kind_tok.start, self.last_end(), kind=kind_tok.value, declarations=declarations)

    def parse_for(self) -> Node:
        start = self.advance().start
        if self.is_word("await"):
            self.advance()
        self.expect("(")
        init = None
        if not self.is_punct(";"):
            self.no_in = True
            if self.tok.kind == "ident" and self.tok.value in ("const", "let", "var"):
                init = self.parse_variable_declaration()
            else:
                init = self.parse_expression()
            self.no_in = False
            if self.is_word("of") or self.is_word("in"):
                each = self.advance().value
                right = self.parse_assignment() if each == "of" else self.parse_expression()
                self.expect(")")
                body = self.parse_statement()
                return Node("ForOfStatement" if each == "of" else "ForInStatement", start, self.last_end(), left=init, right=right, body=body)
        self.expect(";")
        test = None if self.is_punct(";") else self.parse_expression()
        self.expect(";")
        update = None if self.is_punct(")") else self.parse_expression()
        self.expect(")")
        body = self.parse_statement()
        return Node("ForStatement", start, self.last_end(), init=init, test=test, update=update, body=body)

    def parse_function(self, declaration: bool = False) -> Node:
        start = self.tok.start
        if self.is_word("async"):
            self.advance()
        self.advance()  # function
        if self.is_punct("*"):
            raise Unsupported("generator")
        name = None
        if self.tok.kind == "ident" and not self.is_punct("("):
            name = self.parse_binding()
        elif declaration:
            raise JsSyntaxError("função sem nome", start)
        params = self.parse_params()
        body = self.parse_block()
        return Node("FunctionDeclaration" if declaration else "FunctionExpression", start, self.last_end(), id=name, params=params, body=body)

    def parse_params(self) -> list[Node]:
        self.expect("(")
        params = []
        while not self.is_punct(")"):
            if self.is_punct("..."):
                self.advance()
            param = self.parse_binding()
            if self.is_punct("="):
                self.advance()
                self.parse_assignment()
            params.append(param)
            if not self.is_punct(","):
                break
            self.advance()
        self.expect(")")
        return params

    # -- expressões -----------------------------------------------------------------------------------
    def parse_expression(self) -> Node:
        first = self.parse_assignment()
        if not self.is_punct(","):
            return first
        items = [first]
        while self.is_punct(","):
            self.advance()
            items.append(self.parse_assignment())
        return Node("SequenceExpression", first.start, self.last_end(), expressions=items)

    def arrow_ahead(self) -> bool:
        tok = self.tok
        offset = 0
        if self.is_word("async") and not self.peek().nl_before and (self.peek().kind == "ident" or self.is_punct("(", self.peek())):
            offset = 1
            tok = self.peek()
        if tok.kind == "ident" and tok.value not in KEYWORDS - {"of"}:
            return self.is_punct("=>", self.peek(offset + 1))
        if tok.kind == "punct" and tok.value == "(":
            depth = 0
            j = self.i + offset
            while j < len(self.tokens):
                t = self.tokens[j]
                if t.kind == "punct" and t.value in "([{":
                    depth += 1
                elif t.kind == "punct" and t.value in ")]}":
                    depth -= 1
                    if depth == 0:
                        nxt = self.tokens[min(j + 1, len(self.tokens) - 1)]
                        return nxt.kind == "punct" and nxt.value == "=>"
                elif t.kind == "eof":
                    return False
                j += 1
        return False

    def parse_arrow(self) -> Node:
        start = self.tok.start
        if self.is_word("async"):
            self.advance()
        if self.is_punct("("):
            params = self.parse_params()
        else:
            params = [self.parse_binding()]
        self.expect("=>")
        body = self.parse_block() if self.is_punct("{") else self.parse_assignment()
        return Node("ArrowFunctionExpression", start, self.last_end(), params=params, body=body)

    def parse_assignment(self) -> Node:
        if self.arrow_ahead():
            return self.parse_arrow()
        left = self.parse_conditional()
        if self.tok.kind == "punct" and self.tok.value in ASSIGNMENT:
            op = self.advance().value
            right = self.parse_assignment()
            return Node("AssignmentExpression", left.start, self.last_end(), operator=op, left=left, right=right)
        return left

    def parse_conditional(self) -> Node:
        test = self.parse_binary(0)
        if not self.is_punct("?"):
            return test
        self.advance()
        saved, self.no_in = self.no_in, False
        consequent = self.parse_assignment()
        self.no_in = saved
        self.expect(":")
        alternate = self.parse_assignment()
        return Node("ConditionalExpression", test.start, self.last_end(), test=test, consequent=consequent, alternate=alternate)

    def binary_operator(self) -> str | None:
        tok = self.tok
        if tok.kind == "punct" and tok.value in BINARY_PRECEDENCE:
            return tok.value
        if tok.kind == "ident" and tok.value in ("instanceof", "in") and not (tok.value == "in" and self.no_in):
            return tok.value
        return None

    def parse_binary(self, min_prec: int) -> Node:
        left = self.parse_unary()
        while (op := self.binary_operator()) and BINARY_PRECEDENCE[op] > min_prec:
            self.advance()
            right = self.parse_binary(BINARY_PRECEDENCE[op] - (1 if op == "**" else 0))
            left = Node("LogicalExpression" if op in ("&&", "||", "??") else "BinaryExpression",
                        left.start, self.last_end(), operator=op, left=left, right=right)
        return left

    def parse_unary(self) -> Node:
        tok = self.tok
        if (tok.kind == "punct" and tok.value in ("!", "-", "+", "~", "++", "--")) or \
                (tok.kind == "ident" and tok.value in ("typeof", "void", "delete", "await")):
            self.advance()
            argument = self.parse_unary()
            return Node("UnaryExpression", tok.start, self.last_end(), operator=tok.value, argument=argument)
        expression = self.parse_call_member()
        if self.tok.kind == "punct" and self.tok.value in ("++", "--") and not self.tok.nl_before:
            op = self.advance().value
            return Node("UpdateExpression", expression.start, self.last_end(), operator=op, argument=expression)
        return expression

    def parse_arguments(self) -> list[Node]:
        self.expect("(")
        args = []
        while not self.is_punct(")"):
            if self.is_punct("..."):
                start = self.advance().start
                argument = self.parse_assignment()
                args.append(Node("SpreadElement", start, self.last_end(), argument=argument))
            else:
                args.append(self.parse_assignment())
            if not self.is_punct(","):
                break
            self.advance()
        self.expect(")")
        return args

    def parse_property_name(self) -> Node:
        tok = self.advance()
        if tok.kind != "ident":
            raise JsSyntaxError("nome de propriedade esperado", tok.start)
        return Node("Identifier", tok.start, tok.end, name=tok.value)

    def parse_call_member(self) -> Node:
        start = self.tok.start
        if self.is_word("new"):
            self.advance()
            callee = self.parse_primary()
            while self.is_punct(".") or self.is_punct("["):
                callee = self.parse_member_step(callee, start)
            args = self.parse_arguments() if self.is_punct("(") else []
            expression = Node("NewExpression", start, self.last_end(), callee=callee, arguments=args)
        else:
            expression = self.parse_primary()
        while True:
            if self.is_punct(".") or self.is_punct("[") or self.is_punct("?."):
                expression = self.parse_member_step(expression, start)
            elif self.is_punct("("):
                args = self.parse_arguments()
                expression = Node("CallExpression", start, self.last_end(), callee=expression, arguments=args, optional=False)
            elif self.tok.kind == "template" and not self.tok.nl_before:
                self.advance()
                expression = Node("TaggedTemplateExpression", start, self.last_end(), tag=expression)
            else:
                return expression

    def parse_member_step(self, obj: Node, start: int) -> Node:
        optional = self.is_punct("?.")
        self.advance()
        if optional and self.is_punct("("):
            args = self.parse_arguments()
            return Node("CallExpression", start, self.last_end(), callee=obj, arguments=args, optional=True)
        if self.tokens[self.i - 1].value == "[" or (optional and self.is_punct("[")):
            if optional:
                self.advance()
            prop = self.parse_expression()
            self.expect("]")
            return Node("MemberExpression", start, self.last_end(), object=obj, property=prop, computed=True, optional=optional)
        prop = self.parse_property_name()
        return Node("MemberExpression", start, self.last_end(), object=obj, property=prop, computed=False, optional=optional)

    def parse_primary(self) -> Node:
        tok = self.tok
        if tok.kind == "ident":
            if tok.value == "function" or (tok.value == "async" and self.is_word("function", self.peek())):
                return self.parse_function()
            if tok.value in ("class", "yield", "import", "super"):
                raise Unsupported(tok.value)
            if tok.value in KEYWORDS - {"this", "null", "true", "false", "of", "async", "await"}:
                raise JsSyntaxError(f"palavra reservada inesperada {tok.value!r}", tok.start)
            self.advance()
            if tok.value in ("null", "true", "false"):
                return Node("Literal", tok.start, tok.end, value={"null": None, "true": True, "false": False}[tok.value], raw=tok.value)
            if tok.value == "this":
                return Node("ThisExpression", tok.start, tok.end)
            return Node("Identifier", tok.start, tok.end, name=tok.value)
        if tok.kind == "num":
            self.advance()
            return Node("Literal", tok.start, tok.end, value=_number(tok.value), raw=tok.value)
        if tok.kind == "str":
            self.advance()
            return Node("Literal", tok.start, tok.end, value=_string(tok.value), raw=tok.value)
        if tok.kind == "regex":
            self.advance()
            body, _, flags = tok.value[1:].rpartition("/")
            return Node("Literal", tok.start, tok.end, value=None, raw=tok.value, regex={"pattern": body, "flags": flags})
        if tok.kind == "template":
            self.advance()
            return Node("TemplateLiteral", tok.start, tok.end, raw=tok.value)
        if self.is_punct("("):
            self.advance()
            saved, self.no_in = self.no_in, False
            inner = self.parse_expression()
            self.no_in = saved
            self.expect(")")
            return Node("ParenthesizedExpression", tok.start, self.last_end(), expression=inner)
        if self.is_punct("["):
            self.advance()
            elements: list[Node | None] = []
            while not self.is_punct("]"):
                if self.is_punct(","):
                    self.advance()
                    elements.append(None)
                    continue
                if self.is_punct("..."):
                    s = self.advance().start
                    arg = self.parse_assignment()
                    elements.append(Node("SpreadElement", s, self.last_end(), argument=arg))
                else:
                    elements.append(self.parse_assignment())
                if not self.is_punct(","):
                    break
                self.advance()
            self.expect("]")
            return Node("ArrayExpression", tok.start, self.last_end(), elements=elements)
        if self.is_punct("{"):
            return self.parse_object()
        if tok.kind == "eof":
            raise JsSyntaxError("fim inesperado", tok.start)
        raise JsSyntaxError(f"token inesperado {tok.value!r}", tok.start)

    def parse_object(self) -> Node:
        start = self.expect("{").start
        properties = []
        while not self.is_punct("}"):
            p_start = self.tok.start
            if self.is_punct("..."):
                self.advance()
                arg = self.parse_assignment()
                properties.append(Node("SpreadElement", p_start, self.last_end(), argument=arg))
            else:
                computed = False
                tok = self.tok
                if self.is_punct("["):
                    self.advance()
                    key = self.parse_assignment()
                    self.expect("]")
                    computed = True
                elif tok.kind in ("ident", "str", "num"):
                    self.advance()
                    key = Node("Identifier", tok.start, tok.end, name=tok.value) if tok.kind == "ident" else \
                        Node("Literal", tok.start, tok.end, value=_string(tok.value) if tok.kind == "str" else _number(tok.value), raw=tok.value)
                else:
                    raise JsSyntaxError("chave de objeto esperada", tok.start)
                if self.is_punct(":"):
                    self.advance()
                    value = self.parse_assignment()
                    shorthand = False
                elif self.is_punct("("):
                    params = self.parse_params()
                    body = self.parse_block()
                    value = Node("FunctionExpression", key.start, self.last_end(), id=None, params=params, body=body)
                    shorthand = False
                elif tok.kind == "ident" and not computed:
                    value = key
                    shorthand = True
                else:
                    raise JsSyntaxError("':' esperado", self.tok.start)
                properties.append(Node("Property", p_start, self.last_end(), key=key, value=value, computed=computed, shorthand=shorthand))
            if not self.is_punct(","):
                break
            self.advance()
        self.expect("}")
        return Node("ObjectExpression", start, self.last_end(), properties=properties)


def _number(raw: str) -> float | int:
    text = raw.replace("_", "")
    try:
        if text.endswith("n"):
            return int(text[:-1], 0)
        if text[:2].lower() in ("0x", "0b", "0o"):
            return int(text, 0)
        value = float(text)
        return int(value) if value.is_integer() and "." not in text and "e" not in text.lower() else value
    except ValueError as exc:
        raise JsSyntaxError("número inválido", 0) from exc


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


def _string(raw: str) -> str:
    body = raw[1:-1]
    if "\\" not in body:
        return body
    out = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt == "u" and i + 5 < len(body):
                try:
                    out.append(chr(int(body[i + 2:i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse(src: str) -> Node:
    return Parser(src).parse_program()


def key_name(prop: Node) -> str | None:
    """Nome estático da chave de uma Property (None se computada)."""
    if prop.type != "Property" or prop.computed:
        return None
    key = prop.key
    return key.name if key.type == "Identifier" else str(key.value)
