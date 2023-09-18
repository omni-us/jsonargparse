import ast
import inspect
import logging
import sys
import textwrap
from collections import namedtuple
from copy import deepcopy
from dataclasses import is_dataclass
from importlib import import_module
from typing import Any, Dict, ForwardRef, FrozenSet, List, Optional, Set, Tuple, Type, Union, get_type_hints

from ._optionals import typing_extensions_import
from ._typehints import mapping_origin_types, sequence_origin_types, tuple_set_origin_types
from ._util import get_typehint_origin

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


class TypeCheckingVisitor(ast.NodeVisitor):
    type_checking_names: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "typing":
                name = ast.dump(
                    ast.Attribute(
                        value=ast.Name(id=alias.asname or "typing", ctx=ast.Load()),
                        attr="TYPE_CHECKING",
                        ctx=ast.Load(),
                    )
                )
                self.type_checking_names.append(name)
                break

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "typing":
            for alias in node.names:
                if alias.name == "TYPE_CHECKING":
                    name = ast.dump(ast.Name(id=alias.asname or "TYPE_CHECKING", ctx=ast.Load()))
                    self.type_checking_names.append(name)
                    break

    def visit_If(self, node: ast.If) -> None:
        if (
            isinstance(node.test, (ast.Name, ast.Attribute))
            and any(ast.dump(node.test) == n for n in self.type_checking_names)
        ) or (
            isinstance(node.test, ast.BoolOp)
            and isinstance(node.test.op, (ast.And, ast.Or))
            and any(ast.dump(v) == n for n in self.type_checking_names for v in node.test.values)
        ):
            ast_exec = ast.parse("")
            ast_exec.body = node.body
            try:
                exec(compile(ast_exec, filename="<ast>", mode="exec"), self.aliases, self.aliases)
            except Exception as ex:
                if self.logger:
                    self.logger.debug(f"Failed to execute 'TYPE_CHECKING' block in '{self.module}'", exc_info=ex)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.If, ast.Module)):
            super().generic_visit(node)

    def update_aliases(
        self, module_source: str, module: str, aliases: dict, logger: Optional[logging.Logger] = None
    ) -> None:
        self.module = module
        self.aliases = aliases
        self.logger = logger
        module_tree = ast.parse(module_source)
        self.visit(module_tree)


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
    return exec_vars["___arg_type___"]


def getattr_recursive(obj, attr):
    if "." in attr:
        attr, *attrs = attr.split(".", 1)
        return getattr_recursive(getattr(obj, attr), attrs[0])
    return getattr(obj, attr)


def resolve_forward_refs(arg_type, aliases, logger):
    if isinstance(arg_type, str) and arg_type in aliases:
        arg_type = aliases[arg_type]

    def resolve_subtypes_forward_refs(typehint):
        if has_subtypes(typehint):
            try:
                subtypes = []
                for arg in typehint.__args__:
                    if isinstance(arg, ForwardRef):
                        forward_arg, *forward_args = arg.__forward_arg__.split(".", 1)
                        if forward_arg in aliases:
                            arg = aliases[forward_arg]
                            if forward_args:
                                arg = getattr_recursive(arg, forward_args[0])
                        else:
                            raise NameError(f"Name '{forward_arg}' is not defined")
                    else:
                        arg = resolve_subtypes_forward_refs(arg)
                    subtypes.append(arg)
                if subtypes != list(typehint.__args__):
                    typehint_origin = get_typehint_origin(typehint)
                    if sys.version_info < (3, 10):
                        if typehint_origin in sequence_origin_types:
                            typehint_origin = List
                        elif typehint_origin in tuple_set_origin_types:
                            typehint_origin = Tuple
                        elif typehint_origin in mapping_origin_types:
                            typehint_origin = Dict
                    typehint = typehint_origin[tuple(subtypes)]
            except Exception as ex:
                if logger:
                    logger.debug(f"Failed to resolve forward refs in {typehint}", exc_info=ex)
        return typehint

    return resolve_subtypes_forward_refs(arg_type)


def has_subtypes(typehint):
    typehint_origin = get_typehint_origin(typehint)
    return (
        typehint_origin == Union
        or typehint_origin in sequence_origin_types
        or typehint_origin in tuple_set_origin_types
        or typehint_origin in mapping_origin_types
    )


def type_requires_eval(typehint):
    if has_subtypes(typehint):
        return any(type_requires_eval(a) for a in getattr(typehint, "__args__", []))
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
        assert isinstance(node, (ast.FunctionDef, ast.ClassDef))
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

    module_source = inspect.getsource(sys.modules[obj.__module__]) if obj.__module__ in sys.modules else ""
    if "TYPE_CHECKING" in module_source:
        TypeCheckingVisitor().update_aliases(module_source, obj.__module__, aliases, logger)

    if isinstance(node, ast.FunctionDef):
        arg_asts = [(a.arg, a.annotation) for a in node.args.args + node.args.kwonlyargs]
    else:
        arg_asts = [(a.target.id, a.annotation) for a in node.body if isinstance(a, ast.AnnAssign)]  # type: ignore

    for name, annotation in arg_asts:
        if annotation and (name not in types or type_requires_eval(types[name])):
            try:
                if isinstance(annotation, ast.Constant) and annotation.value in aliases:
                    types[name] = aliases[annotation.value]
                else:
                    arg_type = get_arg_type(annotation, aliases)
                    types[name] = resolve_forward_refs(arg_type, aliases, logger)
            except Exception as ex3:
                types[name] = ex3

    if all(isinstance(t, Exception) for t in types.values()):
        raise ex or next(iter(types.values()))

    return types


def evaluate_postponed_annotations(params, component, parent, logger):
    if not (params and any(type_requires_eval(p.annotation) for p in params)):
        return
    try:
        if (
            is_dataclass(parent)
            and component.__name__ == "__init__"
            and not component.__qualname__.startswith(parent.__name__ + ".")
        ):
            types = get_types(parent, logger)
        else:
            types = get_types(component, logger)
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
