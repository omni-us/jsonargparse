import ast
import inspect
import logging
import sys
import textwrap
from collections import namedtuple
from copy import deepcopy
from importlib import import_module
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Type, Union, get_type_hints

if sys.version_info[:2] > (3, 6):
    from typing import ForwardRef

from ._optionals import typing_extensions_import

var_map = namedtuple("var_map", "name value")
none_map = var_map(name="NoneType", value=type(None))
union_map = var_map(name="Union", value=Union)
pep585_map = {
    "dict": var_map(name="Dict", value=Dict),
    "frozenset": var_map(name="FrozenSet", value=FrozenSet),
    "list": var_map(name="List", value=List),
    "set": var_map(name="Set", value=Set),
    "tuple": var_map(name="Tuple", value=Tuple),
    "type": var_map(name="Type", value=Type),
}


class BackportTypeHints(ast.NodeTransformer):
    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        if isinstance(node.value, ast.Name) and node.value.id in pep585_map:
            value = self.new_name_load(pep585_map[node.value.id])
        else:
            value = node.value  # type: ignore
        return ast.Subscript(
            value=value,
            slice=self.visit(node.slice),
            ctx=ast.Load(),
        )

    def visit_Constant(self, node: ast.Constant) -> Union[ast.Constant, ast.Name]:
        if node.value is None:
            return self.new_name_load(none_map)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> Union[ast.BinOp, ast.Subscript]:
        out_node: Union[ast.BinOp, ast.Subscript] = node
        if isinstance(node.op, ast.BitOr):
            elts: list = []
            self.append_union_elts(node.left, elts)
            self.append_union_elts(node.right, elts)
            out_node = ast.Subscript(
                value=self.new_name_load(union_map),
                slice=ast.Index(
                    value=ast.Tuple(elts=elts, ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            )
        return out_node

    def append_union_elts(self, node: ast.AST, elts: list) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            self.append_union_elts(node.left, elts)
            self.append_union_elts(node.right, elts)
        else:
            elts.append(self.visit(node))

    def new_name_load(self, var: var_map) -> ast.Name:
        name = f"_{self.__class__.__name__}_{var.name}"
        self.exec_vars[name] = var.value
        return ast.Name(id=name, ctx=ast.Load())

    def backport(self, input_ast: ast.AST, exec_vars: dict) -> ast.AST:
        typing = __import__("typing")
        for key, value in exec_vars.items():
            if getattr(value, "__module__", "") == "collections.abc":
                if hasattr(typing, key):
                    exec_vars[key] = getattr(typing, key)
        self.exec_vars = exec_vars
        backport_ast = self.visit(deepcopy(input_ast))
        return ast.fix_missing_locations(backport_ast)


class NamesVisitor(ast.NodeVisitor):
    def visit_Name(self, node: ast.Name) -> None:
        self.names_found.append(node.id)

    def find(self, node: ast.AST) -> list:
        from ._util import unique

        self.names_found: List[str] = []
        self.visit(node)
        self.names_found = unique(self.names_found)
        return self.names_found


def get_arg_type(arg_ast, aliases):
    type_ast = ast.parse("___arg_type___ = 0")
    type_ast.body[0].value = arg_ast
    exec_vars = {}
    bad_aliases = {}
    add_asts = False
    for name in NamesVisitor().find(arg_ast):
        value = aliases[name]
        if isinstance(value, tuple):
            value = value[1]
        if isinstance(value, Exception):
            bad_aliases[name] = value
        elif isinstance(value, ast.AST):
            add_asts = True
        else:
            exec_vars[name] = value
    if add_asts:
        body = []
        for name, (_, value) in aliases.items():
            if isinstance(value, ast.AST):
                body.append(ast.fix_missing_locations(value))
            elif not isinstance(value, Exception):
                exec_vars[name] = value
        type_ast.body = body + type_ast.body
        if "TypeAlias" not in exec_vars:
            type_alias = typing_extensions_import("TypeAlias")
            if type_alias:
                exec_vars["TypeAlias"] = type_alias
    if sys.version_info < (3, 10):
        backporter = BackportTypeHints()
        type_ast = backporter.backport(type_ast, exec_vars)
    try:
        exec(compile(type_ast, filename="<ast>", mode="exec"), exec_vars, exec_vars)
    except NameError as ex:
        ex_from = None
        for name, alias_exception in bad_aliases.items():
            if str(ex) == f"name '{name}' is not defined":
                ex_from = alias_exception
                break
        raise ex from ex_from
    arg_type = exec_vars["___arg_type___"]
    if isinstance(arg_type, str) and arg_type in aliases:
        arg_type = aliases[arg_type]
    return arg_type


def type_requires_eval(typehint):
    return isinstance(typehint, (str, ForwardRef))


def get_types(obj: Any, logger: Optional[logging.Logger] = None) -> dict:
    global_vars = vars(import_module(obj.__module__))
    try:
        types = get_type_hints(obj, global_vars)
    except Exception as ex1:
        types = ex1  # type: ignore

    if isinstance(types, dict) and all(not type_requires_eval(t) for t in types.values()):
        return types

    try:
        source = textwrap.dedent(inspect.getsource(obj))
        tree = ast.parse(source)
        assert isinstance(tree, ast.Module) and len(tree.body) == 1
        node = tree.body[0]
        assert isinstance(node, ast.FunctionDef)
    except Exception as ex2:
        if isinstance(types, Exception):
            if logger:
                logger.debug(f"Failed to parse to source code for {obj}", exc_info=ex2)
            raise type(types)(f"{repr(types)} + {repr(ex2)}") from ex2  # type: ignore
        return types

    aliases = __builtins__.copy()  # type: ignore
    aliases.update(global_vars)
    ex = None
    if isinstance(types, Exception):
        ex = types
        types = {}

    for arg_ast in node.args.args + node.args.kwonlyargs:
        name = arg_ast.arg
        if arg_ast.annotation and (name not in types or type_requires_eval(types[name])):
            try:
                if isinstance(arg_ast.annotation, ast.Constant) and arg_ast.annotation.value in aliases:
                    types[name] = aliases[arg_ast.annotation.value]
                else:
                    types[name] = get_arg_type(arg_ast.annotation, aliases)
            except Exception as ex3:
                types[name] = ex3

    if all(isinstance(t, Exception) for t in types.values()):
        raise ex or next(iter(types.values()))

    return types


def evaluate_postponed_annotations(params, component, logger):
    if sys.version_info[:2] == (3, 6) or not (params and any(type_requires_eval(p.annotation) for p in params)):
        return
    try:
        if sys.version_info < (3, 10):
            types = get_types(component, logger)
        else:
            types = get_type_hints(component)
    except Exception as ex:
        logger.debug(f"Unable to evaluate types for {component}", exc_info=ex)
        return
    for param in params:
        if param.name in types:
            param_type = types[param.name]
            if isinstance(param_type, Exception):
                logger.debug(f"Unable to evaluate type of {param.name} from {component}", exc_info=param_type)
                continue
            param.annotation = param_type
