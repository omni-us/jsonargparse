import ast
import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .optionals import (
    import_typeshed_client,
    typeshed_client_support,
    typing_extensions_import,
)
from .util import import_object, unique

if TYPE_CHECKING:  # pragma: no cover
    import typeshed_client as tc
else:
    tc = import_typeshed_client()


kinds = inspect._ParameterKind


def import_module(name: str):
    try:
        if '.' in name:
            return import_object(name)
        return __import__(name)
    except Exception:
        return None


class NamesVisitor(ast.NodeVisitor):

    def visit_Name(self, node: ast.Name) -> None:
        self.names_found.append(node.id)

    def find(self, node: ast.AST) -> list:
        self.names_found: List[str] = []
        self.visit(node)
        self.names_found = unique(self.names_found)
        return self.names_found


class ImportsVisitor(ast.NodeVisitor):

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            module_path = self.module_path[:-node.level]
            if node.module:
                module_path.append(node.module)
            node.module = '.'.join(module_path)
            node.level = 0
        for alias in node.names:
            self.imports_found[alias.asname or alias.name] = (node.module, alias.name)

    def find(self, node: ast.AST, module_path: str) -> Dict[str, Tuple[Optional[str], str]]:
        self.module_path = module_path.split('.')
        self.imports_found: Dict[str, Tuple[Optional[str], str]] = {}
        self.visit(node)
        return self.imports_found


def ast_annassign_to_assign(node: ast.AnnAssign) -> ast.Assign:
    return ast.Assign(targets=[node.target], value=node.value, type_ignores=[], lineno=1, end_lineno=1)


class AssignsVisitor(ast.NodeVisitor):

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if hasattr(target, 'id'):
                self.assigns_found[target.id] = node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if hasattr(node.target, 'id'):
            self.assigns_found[node.target.id] = ast_annassign_to_assign(node)

    def find(self, node: ast.AST) -> Dict[str, ast.Assign]:
        self.assigns_found: Dict[str, ast.Assign] = {}
        self.visit(node)
        return self.assigns_found


stubs_resolver = None


def get_stubs_resolver():
    global stubs_resolver
    if not stubs_resolver:
        search_path = [Path(p) for p in sys.path]
        search_context = tc.get_search_context(search_path=search_path)
        stubs_resolver = StubsResolver(search_context=search_context)
    return stubs_resolver


def ast_get_class_method(node: ast.AST, method_name: str) -> Optional[ast.FunctionDef]:
    method_ast = None
    if isinstance(node, ast.ClassDef):
        for elem in node.body:
            if isinstance(elem, ast.FunctionDef) and elem.name == method_name:
                method_ast = elem
                break
    return method_ast


def get_mro_method_parent(parent, method_name):
    while hasattr(parent, '__dict__') and method_name not in parent.__dict__:
        try:
            parent = inspect.getmro(parent)[1]
        except IndexError:
            parent = None
    return None if parent is object else parent


class StubsResolver(tc.Resolver):

    def __init__(self, search_context = None) -> None:
        super().__init__(search_context)
        self._module_ast_cache: Dict[str, Optional[ast.AST]] = {}
        self._module_assigns_cache: Dict[str, Dict[str, ast.Assign]] = {}
        self._module_imports_cache: Dict[str, Dict[str, Tuple[Optional[str], str]]] = {}

    def get_imported_info(self, name: str) -> Optional[tc.ImportedInfo]:
        resolved = super().get_fully_qualified_name(name)
        imported_info = None
        if isinstance(resolved, tc.ImportedInfo):
            imported_info = resolved
        elif isinstance(resolved, tc.NameInfo):
            source_module = tc.ModulePath(tuple(name.split('.')[:-1]))
            imported_info = tc.ImportedInfo(source_module=source_module, info=resolved)
        return imported_info

    def get_component_imported_info(self, component, parent) -> Optional[tc.ImportedInfo]:
        if not parent and inspect.ismethod(component):
            parent = type(component.__self__)
            component = getattr(parent, component.__name__)
        if not parent:
            return self.get_imported_info(f'{component.__module__}.{component.__name__}')
        parent = get_mro_method_parent(parent, component.__name__)
        stub_import = parent and self.get_imported_info(f'{parent.__module__}.{parent.__name__}')
        if stub_import and isinstance(stub_import.info.ast, ast.AST):
            method_ast = ast_get_class_method(stub_import.info.ast, component.__name__)
            if method_ast is not None:
                name_info = tc.NameInfo(name=component.__qualname__, is_exported=False, ast=method_ast)
                stub_import = tc.ImportedInfo(source_module=stub_import.source_module, info=name_info)
        return stub_import

    def get_aliases(self, imported_info: tc.ImportedInfo):
        aliases: Dict[str, Tuple[str, Any]] = {}
        self.add_import_aliases(aliases, imported_info)
        return aliases

    def get_module_stub_ast(self, module_path: str):
        if module_path not in self._module_ast_cache:
            self._module_ast_cache[module_path] = tc.get_stub_ast(module_path, search_context=self.ctx)
        return self._module_ast_cache[module_path]

    def get_module_stub_assigns(self, module_path: str):
        if module_path not in self._module_assigns_cache:
            module_ast = self.get_module_stub_ast(module_path)
            self._module_assigns_cache[module_path] = AssignsVisitor().find(module_ast)
        return self._module_assigns_cache[module_path]

    def get_module_stub_imports(self, module_path: str):
        if module_path not in self._module_imports_cache:
            module_ast = self.get_module_stub_ast(module_path)
            self._module_imports_cache[module_path] = ImportsVisitor().find(module_ast, module_path)
        return self._module_imports_cache[module_path]

    def add_import_aliases(self, aliases, stub_import: tc.ImportedInfo):
        module_path = '.'.join(stub_import.source_module)
        module = import_module(module_path)
        stub_ast: Optional[ast.AST] = None
        if isinstance(stub_import.info.ast, (ast.Assign, ast.AnnAssign)):
            stub_ast = stub_import.info.ast.value
        elif isinstance(stub_import.info.ast, ast.AST):
            stub_ast = stub_import.info.ast
        if stub_ast:
            self.add_module_aliases(aliases, module_path, module, stub_ast)
        return module_path, stub_import.info.ast

    def add_module_aliases(self, aliases, module_path, module, node):
        names = NamesVisitor().find(node) if node else []
        for name in names:
            if alias_already_added(aliases, name, module_path):
                continue
            source = module_path
            value = inspect._empty
            if name in __builtins__:
                source = '__builtins__'
                value = __builtins__[name]
            elif hasattr(module, name):
                value = getattr(module, name)
            elif name in self.get_module_stub_assigns(module_path):
                value = self.get_module_stub_assigns(module_path)[name]
                self.add_module_aliases(aliases, module_path, module, value.value)
            elif name in self.get_module_stub_imports(module_path):
                imported_module_path, imported_name = self.get_module_stub_imports(module_path)[name]
                imported_module = import_module(imported_module_path)
                if hasattr(imported_module, imported_name):
                    source = imported_module_path
                    value = getattr(imported_module, imported_name)
                else:
                    stub_import = self.get_imported_info(f'{imported_module_path}.{imported_name}')
                    source, value = self.add_import_aliases(aliases, stub_import)
            else:
                value = NotImplementedError(f'{name!r} from {module_path!r} not in builtins, module or stub')
            if alias_already_added(aliases, name, source):
                continue
            if not alias_is_unique(aliases, name, source, value):
                value = NotImplementedError(
                    f'non-unique alias {name!r}: {aliases[name][1]} ({aliases[name][0]}) vs {value} ({source})'
                )
            aliases[name] = (source, value)


def alias_already_added(aliases, name, source):
    return name in aliases and aliases[name][0] in {'__builtins__', source}


def alias_is_unique(aliases, name, source, value):
    if name in aliases:
        src, val = aliases[name]
        if src != source:
            return val is value
    return True


def get_arg_type(arg_ast, aliases):
    type_ast = ast.parse('___arg_type___ = 0')
    type_ast.body[0].value = arg_ast.annotation
    exec_vars = {}
    bad_aliases = {}
    add_asts = False
    for name in NamesVisitor().find(arg_ast.annotation):
        _, value = aliases[name]
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
        if 'TypeAlias' not in exec_vars:
            type_alias = typing_extensions_import('TypeAlias')
            if type_alias:
                exec_vars['TypeAlias'] = type_alias
    try:
        exec(compile(type_ast, filename="<ast>", mode="exec"), exec_vars, exec_vars)
    except NameError as ex:
        for name, alias_exception in bad_aliases.items():
            if str(ex) == f"name '{name}' is not defined":
                raise NameError(str(alias_exception)) from ex
        raise ex
    return exec_vars['___arg_type___']


def get_stub_types(params, component, parent, logger) -> Optional[Dict[str, Any]]:
    if not typeshed_client_support:
        return None
    missing_types = {
        p.name: n for n, p in enumerate(params)
        if p.kind not in {kinds.VAR_POSITIONAL, kinds.VAR_KEYWORD} and p.annotation == inspect._empty
    }
    if not missing_types:
        return None
    resolver = get_stubs_resolver()
    stub_import = resolver.get_component_imported_info(component, parent)
    if not stub_import:
        return None
    known_params = {p.name for p in params}
    aliases = resolver.get_aliases(stub_import)
    arg_asts = stub_import.info.ast.args.args + stub_import.info.ast.args.kwonlyargs
    types = {}
    for arg_ast in arg_asts[1:] if parent else arg_asts:
        name = arg_ast.arg
        if arg_ast.annotation and (name in missing_types or name not in known_params):
            try:
                types[name] = get_arg_type(arg_ast, aliases)
            except Exception as ex:
                logger.debug(f'Failed to use type stub for parameter {name}', exc_info=ex)
                continue
    return types
