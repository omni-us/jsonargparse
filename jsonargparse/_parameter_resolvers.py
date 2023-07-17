import ast
import inspect
import logging
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from ._backports import evaluate_postponed_annotations
from ._common import is_dataclass_like, is_subclass
from ._optionals import parse_docs
from ._stubs_resolver import get_stub_types
from ._util import (
    ClassFromFunctionBase,
    LoggerProperty,
    get_import_path,
    iter_to_set_str,
    parse_logger,
    unique,
)


@dataclass
class ParamData:
    name: str
    annotation: Any
    default: Any = inspect._empty
    kind: Optional[inspect._ParameterKind] = None
    doc: Optional[str] = None
    component: Optional[Union[Callable, Type, Tuple]] = None
    parent: Optional[Union[Type, Tuple]] = None
    origin: Optional[Union[str, Tuple]] = None


ParamList = List[ParamData]
parameter_attributes = [s[1:] for s in inspect.Parameter.__slots__]  # type: ignore
kinds = inspect._ParameterKind
ast_assign_type: Tuple[Type[ast.AST], ...] = (ast.AnnAssign, ast.Assign)
param_kwargs_get = "**.get()"
param_kwargs_pop = "**.pop()"


class SourceNotAvailable(Exception):
    "Raised when the source code for some component is not available."


class UnknownDefault:
    def __init__(self, resolver: str, data: Any = inspect._empty) -> None:
        self.resolver = resolver
        self.data = data

    def __repr__(self) -> str:
        value = f"{type(self).__name__.replace('Default', '')}<{self.resolver}>"
        if self.data != inspect._empty:
            value = f"{value} {self.data}"
        return value


class ConditionalDefault(UnknownDefault):
    def __init__(self, resolver: str, data: Any) -> None:
        super().__init__(resolver, iter_to_set_str(data, sep=", "))


def get_parameter_origins(component, parent) -> Optional[str]:
    if isinstance(component, tuple):
        assert parent is None or len(component) == len(parent)
        return iter_to_set_str(get_parameter_origins(c, parent[n] if parent else None) for n, c in enumerate(component))
    if parent:
        return f"{get_import_path(parent)}.{component.__name__}"
    return get_import_path(component)


def is_staticmethod(attr) -> bool:
    return isinstance(attr, staticmethod)


def is_method(attr) -> bool:
    return (inspect.isfunction(attr) or attr.__class__.__name__ == "cython_function_or_method") and not is_staticmethod(
        attr
    )


def is_property(attr) -> bool:
    return isinstance(attr, property)


def is_method_or_property(attr) -> bool:
    return is_method(attr) or is_property(attr)


def is_classmethod(parent, component) -> bool:
    return parent and isinstance(inspect.getattr_static(parent, component.__name__), classmethod)


def is_lambda(value: Any) -> bool:
    return callable(value) and value.__name__ == "<lambda>"


def ast_str(node):
    return getattr(ast, "unparse", ast.dump)(node)


def ast_variable_load(name):
    return ast.Name(id=name, ctx=ast.Load())


def ast_attribute_load(container, name):
    return ast.Attribute(value=ast.Name(id=container, ctx=ast.Load()), attr=name, ctx=ast.Load())


def ast_is_assign_with_value(node, value) -> bool:
    return isinstance(node, ast_assign_type) and ast.dump(node.value) == ast.dump(value)


def ast_get_assign_targets(node):
    return node.targets if isinstance(node, ast.Assign) else [node.target]


def ast_is_not(node) -> bool:
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)


dict_ast = ast.dump(ast_variable_load("dict"))


def ast_is_dict_assign(node):
    return isinstance(node, ast_assign_type) and (
        isinstance(node.value, ast.Dict) or (isinstance(node.value, ast.Call) and ast.dump(node.value.func) == dict_ast)
    )


def ast_is_dict_assign_with_value(node, value):
    if ast_is_dict_assign(node) and getattr(node.value, "keywords", None):
        value_dump = ast.dump(value)
        for keyword in [k.value for k in node.value.keywords]:
            if ast.dump(keyword) == value_dump:
                return True
    return False


def ast_is_call_with_value(node, value_dump) -> bool:
    for argtype in ["args", "keywords"]:
        for arg in getattr(node, argtype):
            if isinstance(getattr(arg, "value", None), ast.AST) and ast.dump(arg.value) == value_dump:
                return True
    return False


ast_constant_attr = {
    ast.Constant: "value",
    # python <= 3.7:
    ast.NameConstant: "value",
    ast.Num: "n",
    ast.Str: "s",
}


def ast_is_constant(node):
    return isinstance(node, (ast.Str, ast.Num, ast.NameConstant, ast.Constant))


def ast_get_constant_value(node):
    assert ast_is_constant(node)
    return getattr(node, ast_constant_attr[node.__class__])


def ast_get_name_and_attrs(node) -> List[str]:
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return names[::-1]


def ast_is_kwargs_pop_or_get(node, value_dump) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and value_dump == ast.dump(node.func.value)
        and node.func.attr in {"pop", "get"}
        and len(node.args) == 2
        and isinstance(ast_get_constant_value(node.args[0]), str)
    )


def ast_is_super_call(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
    )


def ast_is_supported_super_call(node, self_name, log_debug) -> bool:
    supported = False
    args = node.func.value.args
    if not args and not node.func.value.keywords:
        supported = True
    elif (
        args
        and len(args) == 2
        and all(isinstance(a, ast.Name) for a in args)
        and self_name == args[1].id
        and not node.func.value.keywords
    ):
        classes, idx = current_mro.get()
        module = inspect.getmodule(classes[idx])
        for offset, cls in enumerate(classes[idx:]):
            if args[0].id == cls.__name__ and cls is getattr(module, cls.__name__, None):
                current_mro.set((classes, idx + offset))
                supported = True
                break
    if not supported:
        log_debug(f"unsupported super parameters: {ast_str(node)}")
    return supported


def ast_is_attr_assign(node, container):
    for target in ast_get_assign_targets(node) if isinstance(node, ast_assign_type) else []:
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == container:
            return target.attr
    return False


def ast_get_call_kwarg_with_value(node, value):
    value_dump = ast.dump(value)
    kwarg = None
    for arg in node.keywords:
        if isinstance(getattr(arg, "value", None), ast.AST) and ast.dump(arg.value) == value_dump:
            kwarg = arg
            break
    return kwarg


def ast_get_call_positional_indexes(node):
    return [n for n, a in enumerate(node.args) if not isinstance(a, ast.Starred)]


def ast_get_call_keyword_names(node):
    return [kw_node.arg for kw_node in node.keywords if kw_node.arg]


def remove_given_parameters(node, params):
    given_args = set(ast_get_call_positional_indexes(node))
    given_kwargs = set(ast_get_call_keyword_names(node))
    params = [p for n, p in enumerate(params) if n not in given_args]
    params = [p for p in params if p.name not in given_kwargs]
    return params


def get_arg_kind_index(params, kind):
    return next((n for n, p in enumerate(params) if p.kind == kind), -1)


def get_signature_parameters_and_indexes(component, parent, logger):
    signature_source = component
    if is_classmethod(parent, component):
        signature_source = component.__func__
    params = list(inspect.signature(signature_source).parameters.values())
    if parent:
        params = params[1:]
    args_idx = get_arg_kind_index(params, kinds.VAR_POSITIONAL)
    kwargs_idx = get_arg_kind_index(params, kinds.VAR_KEYWORD)
    doc_params = parse_docs(component, parent, logger)
    for num, param in enumerate(params):
        params[num] = ParamData(
            doc=doc_params.get(param.name),
            parent=parent,
            component=component,
            **{a: getattr(param, a) for a in parameter_attributes},
        )
    evaluate_postponed_annotations(params, signature_source, logger)
    stubs = get_stub_types(params, signature_source, parent, logger)
    return params, args_idx, kwargs_idx, doc_params, stubs


def add_stub_types(stubs: Optional[Dict[str, Any]], params: ParamList, component) -> None:
    if not stubs:
        return
    for param in params:
        if param.annotation == inspect._empty and param.name in stubs:
            param.annotation = stubs[param.name]
    known_params = {p.name for p in params}
    for name, stub in stubs.items():
        if name not in known_params:
            params.append(
                ParamData(
                    name=name,
                    annotation=stub,
                    default=UnknownDefault("stubs-resolver"),
                    kind=kinds.KEYWORD_ONLY,
                    component=component,
                )
            )


ast_literals = {ast.dump(ast.parse(v, mode="eval").body): partial(ast.literal_eval, v) for v in ["{}", "[]"]}


def get_kwargs_pop_or_get_parameter(node, component, parent, doc_params, log_debug):
    name = ast_get_constant_value(node.args[0])
    if ast_is_constant(node.args[1]):
        default = ast_get_constant_value(node.args[1])
    else:
        default = ast.dump(node.args[1])
        if default in ast_literals:
            default = ast_literals[default]()
        else:
            default = None
            log_debug(f"unsupported kwargs pop/get default: {ast_str(node)}")
    return ParamData(
        name=name,
        annotation=inspect._empty,
        default=default,
        kind=kinds.KEYWORD_ONLY,
        doc=doc_params.get(name),
        parent=parent,
        component=component,
        origin=param_kwargs_get if node.func.attr == "get" else param_kwargs_pop,
    )


def is_param_subclass_instance_default(param: ParamData) -> bool:
    if is_dataclass_like(type(param.default)):
        return False
    from ._typehints import ActionTypeHint, get_subclass_types

    class_types = get_subclass_types(param.annotation)
    return (class_types and isinstance(param.default, class_types)) or (
        is_lambda(param.default)
        and ActionTypeHint.is_callable_typehint(param.annotation, all_subtypes=False)
        and param.annotation.__args__
        and ActionTypeHint.is_subclass_typehint(param.annotation.__args__[-1], all_subtypes=False)
    )


def split_args_and_kwargs(params: ParamList) -> Tuple[ParamList, ParamList]:
    args = [p for p in params if p.kind is kinds.POSITIONAL_ONLY]
    kwargs = [p for p in params if p.kind in {kinds.KEYWORD_ONLY, kinds.POSITIONAL_OR_KEYWORD}]
    return args, kwargs


def replace_args_and_kwargs(params: ParamList, args: ParamList, kwargs: ParamList) -> ParamList:
    args_idx = get_arg_kind_index(params, kinds.VAR_POSITIONAL)
    kwargs_idx = get_arg_kind_index(params, kinds.VAR_KEYWORD)
    if args_idx >= 0:
        params = params[:args_idx] + args + params[args_idx + 1 :]
        if kwargs_idx >= 0:
            kwargs_idx += len(args) - 1
    if kwargs_idx >= 0:
        existing_names = set(p.name for p in params[:kwargs_idx] + params[kwargs_idx + 1 :])
        kwargs = [p for p in kwargs if p.name not in existing_names]
        params = params[:kwargs_idx] + kwargs + params[kwargs_idx + 1 :]
    return params


def group_parameters(params_list: List[ParamList]) -> ParamList:
    if len(params_list) == 1:
        for param in params_list[0]:
            if not isinstance(param.origin, tuple):
                param.origin = None
        return params_list[0]
    grouped = []
    params_count = 0
    params_skip = set()
    params_dict = defaultdict(lambda: [])
    for params in params_list:
        if params[0].origin not in {param_kwargs_get, param_kwargs_pop}:
            params_count += 1
        for param in params:
            if param.name not in params_skip and param.kind != kinds.POSITIONAL_ONLY:
                params_dict[param.name].append(param)
                if param.origin == param_kwargs_pop:
                    params_skip.add(param.name)
    for params in params_dict.values():
        gparam = params[0]
        types = unique(p.annotation for p in params if p.annotation is not inspect._empty)
        defaults = unique(p.default for p in params if p.default is not inspect._empty)
        if len(params) >= params_count and len(types) <= 1 and len(defaults) <= 1:
            gparam.origin = None
        else:
            gparam.parent = tuple(p.parent for p in params)
            gparam.component = tuple(p.component for p in params)
            gparam.origin = tuple(p.origin for p in params)
            gparam.default = ConditionalDefault(
                "ast-resolver",
                (p.default for p in params) if len(defaults) > 1 else defaults,
            )
            if len(types) > 1:
                gparam.annotation = Union[tuple(types)] if types else inspect._empty
        docs = [p.doc for p in params if p.doc]
        gparam.doc = docs[0] if docs else None
        grouped.append(gparam)
    return grouped


def has_dunder_new_method(cls, attr_name):
    classes = inspect.getmro(cls)[1:]
    return (
        attr_name == "__init__"
        and cls.__new__ is not object.__new__
        and not any(cls.__new__ is c.__new__ for c in classes)
    )


current_mro: ContextVar = ContextVar("current_mro", default=(None, None))


@contextmanager
def mro_context(parent):
    token = None
    if parent:
        classes, idx = current_mro.get()
        if not classes or classes[idx] is not parent:
            classes = [c for c in inspect.getmro(parent) if c is not object]
            token = current_mro.set((classes, 0))
    try:
        yield
    finally:
        if token:
            current_mro.reset(token)


def get_mro_parameters(method_name, get_parameters_fn, logger):
    classes, idx = current_mro.get()
    for num, cls in enumerate(classes[idx + 1 :], start=idx + 1):
        method = getattr(cls, method_name, None)
        remainder = classes[num + 1 :] + [object]
        if method and not any(method is getattr(c, method_name, None) for c in remainder):
            current_mro.set((classes, num))
            return get_parameters_fn(cls, method, logger=logger)
    return []


def get_component_and_parent(
    function_or_class: Union[Callable, Type],
    method_or_property: Optional[Union[str, Callable]] = None,
):
    if is_subclass(function_or_class, ClassFromFunctionBase) and method_or_property in {None, "__init__"}:
        function_or_class = function_or_class.wrapped_function  # type: ignore
        method_or_property = None
    elif inspect.isclass(function_or_class) and method_or_property is None:
        method_or_property = "__init__"
    elif method_or_property and not isinstance(method_or_property, str):
        method_or_property = method_or_property.__name__
    parent = component = None
    if method_or_property:
        attr = inspect.getattr_static(function_or_class, method_or_property)
        if is_staticmethod(attr):
            component = getattr(function_or_class, method_or_property)
            return component, parent, method_or_property
        parent = function_or_class
        if has_dunder_new_method(function_or_class, method_or_property):
            component = getattr(function_or_class, "__new__")
        elif is_method(attr):
            component = attr
        elif is_property(attr):
            component = attr.fget
        elif isinstance(attr, classmethod):
            component = getattr(function_or_class, method_or_property)
        elif attr is not object.__init__:
            raise ValueError(
                f"Invalid or unsupported input: class={function_or_class}, method_or_property={method_or_property}"
            )
    else:
        if not callable(function_or_class):
            raise ValueError(f"Non-callable input: function={function_or_class}")
        component = function_or_class
    return component, parent, method_or_property


class ParametersVisitor(LoggerProperty, ast.NodeVisitor):
    def __init__(
        self,
        function_or_class: Union[Callable, Type],
        method_or_property: Optional[Union[str, Callable]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.component, self.parent, _ = get_component_and_parent(function_or_class, method_or_property)

    def log_debug(self, message) -> None:
        self.logger.debug(f"AST resolver: {message}")

    def parse_source_tree(self):
        """Parses the component's AST and sets the component and parent nodes."""
        if hasattr(self, "component_node"):
            return
        try:
            source = textwrap.dedent(inspect.getsource(self.component))
            tree = ast.parse(source)
            assert isinstance(tree, ast.Module) and len(tree.body) == 1
            self.component_node = tree.body[0]
            self.self_name = self.component_node.args.args[0].arg if self.parent else None
        except Exception as ex:
            raise SourceNotAvailable(f"Problems getting source code for {self.component}: {ex}") from ex

    def visit_Assign(self, node):
        do_generic_visit = True
        for key, value in self.find_values.items():
            if ast_is_assign_with_value(node, value):
                self.values_found.append((key, node))
                do_generic_visit = False
                break
            elif ast_is_dict_assign_with_value(node, value):
                self.values_found.append((key, node))
                do_generic_visit = False
        if do_generic_visit:
            if ast_is_dict_assign(node):
                for target in [deepcopy(t) for t in ast_get_assign_targets(node)]:
                    target.ctx = ast.Load()
                    self.dict_assigns[ast.dump(target)] = node
            else:
                self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit_Assign(node)

    def visit_Call(self, node):
        for key, value in self.find_values.items():
            value_dump = ast.dump(value)
            if ast_is_call_with_value(node, value_dump):
                if isinstance(node.func, ast.Attribute):
                    value_dump = ast.dump(node.func.value)
                    if value_dump in self.dict_assigns:
                        self.values_found.append((key, self.dict_assigns[value_dump]))
                        continue
                self.values_found.append((key, node))
            elif ast_is_kwargs_pop_or_get(node, value_dump):
                self.values_found.append((key, node))
        self.generic_visit(node)

    def visit_If(self, node):
        is_test_not = ast_is_not(node.test)
        test_node = node.test.operand if is_test_not else node.test
        component_globals = self.component.__globals__
        if isinstance(test_node, ast.Name) and test_node.id in component_globals:
            condition = bool(component_globals[test_node.id])
            if is_test_not:
                condition = not condition
            body = node.body if condition else node.orelse
            node = ast.If(test=ast.Constant(value=True), body=body, orelse=[])
        self.generic_visit(node)

    def find_values_usage(self, values):
        self.find_values = values
        self.values_found = []
        self.dict_assigns = {}
        self.visit(self.component_node)
        return self.values_found

    def get_node_component(self, node) -> Optional[Tuple[Type, Optional[str]]]:
        function_or_class = method_or_property = None
        module = inspect.getmodule(self.component)
        if isinstance(node.func, ast.Name):
            if is_classmethod(self.parent, self.component) and node.func.id == self.self_name:
                function_or_class = self.parent
            elif hasattr(module, node.func.id):
                function_or_class = getattr(module, node.func.id)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if self.parent and ast.dump(node.func.value) == ast.dump(ast_variable_load(self.self_name)):
                function_or_class = self.parent
                method_or_property = node.func.attr
            elif hasattr(module, node.func.value.id):
                container = getattr(module, node.func.value.id)
                if inspect.isclass(container):
                    function_or_class = container
                    method_or_property = node.func.attr
                elif hasattr(container, node.func.attr):
                    function_or_class = getattr(container, node.func.attr)
        if not function_or_class:
            self.log_debug(f"not supported: {ast_str(node)}")
            return None
        return function_or_class, method_or_property

    def match_call_that_uses_attr(self, node, attr_name):
        params = None
        if isinstance(node, ast.Call):
            params = []
            value = ast_attribute_load(self.self_name, attr_name)
            kwarg = ast_get_call_kwarg_with_value(node, value)
            if kwarg:
                if kwarg.arg:
                    self.log_debug(f"kwargs attribute given as keyword parameter not supported: {ast_str(node)}")
                else:
                    get_param_args = self.get_node_component(node)
                    if get_param_args:
                        params = get_signature_parameters(*get_param_args, logger=self.logger)
            params = remove_given_parameters(node, params)
        return params

    def replace_param_default_subclass_specs(self, params: List[ParamData]) -> None:
        params = [p for p in params if is_param_subclass_instance_default(p)]
        if params:
            self.parse_source_tree()
            default_nodes = self.get_default_nodes({p.name for p in params})
            assert len(params) == len(default_nodes)
            from ._typehints import get_subclass_types

            for param, default_node in zip(params, default_nodes):
                lambda_default = is_lambda(param.default)
                node = default_node
                num_positionals = 0
                if lambda_default:
                    node = default_node.body
                    num_positionals = len(param.annotation.__args__) - 1
                class_type = self.get_call_class_type(node)
                subclass_types = get_subclass_types(param.annotation)
                if not (class_type and subclass_types and is_subclass(class_type, subclass_types)):
                    continue
                subclass_spec: dict = dict(class_path=get_import_path(class_type), init_args=dict())
                for kwarg in node.keywords:
                    if kwarg.arg and ast_is_constant(kwarg.value):
                        subclass_spec["init_args"][kwarg.arg] = ast_get_constant_value(kwarg.value)
                    else:
                        subclass_spec.clear()
                        break
                if not subclass_spec or len(node.args) - num_positionals > 0:
                    self.log_debug(f"unsupported class instance default: {ast_str(default_node)}")
                elif subclass_spec:
                    if not subclass_spec["init_args"]:
                        del subclass_spec["init_args"]
                    param.default = subclass_spec

    def get_call_class_type(self, node) -> Optional[type]:
        names = ast_get_name_and_attrs(getattr(node, "func", None))
        class_type = self.component.__globals__.get(names[0]) if names else None
        for name in names[1:]:
            class_type = getattr(class_type, name, None)
        return class_type if inspect.isclass(class_type) else None

    def get_default_nodes(self, param_names: set):
        node = self.component_node.args
        arg_nodes = getattr(node, "posonlyargs", []) + node.args
        default_nodes = [None] * (len(arg_nodes) - len(node.defaults)) + node.defaults
        default_nodes = [d for n, d in enumerate(default_nodes) if arg_nodes[n].arg in param_names]
        return default_nodes

    def get_parameters_args_and_kwargs(self) -> Tuple[ParamList, ParamList]:
        self.parse_source_tree()
        args_name = getattr(self.component_node.args.vararg, "arg", None)
        kwargs_name = getattr(self.component_node.args.kwarg, "arg", None)
        values_to_find = {}
        if args_name:
            values_to_find[args_name] = ast_variable_load(args_name)
        if kwargs_name:
            values_to_find[kwargs_name] = ast_variable_load(kwargs_name)

        values_found = self.find_values_usage(values_to_find)
        if not values_found:
            return [], []

        params_list = []
        kwargs_value = kwargs_name and values_to_find[kwargs_name]
        kwargs_value_dump = kwargs_value and ast.dump(kwargs_value)
        for node in [v for k, v in values_found if k == kwargs_name]:
            if isinstance(node, ast.Call):
                if ast_is_kwargs_pop_or_get(node, kwargs_value_dump):
                    param = get_kwargs_pop_or_get_parameter(
                        node, self.component, self.parent, self.doc_params, self.log_debug
                    )
                    params_list.append([param])
                    continue
                kwarg = ast_get_call_kwarg_with_value(node, kwargs_value)
                params = []
                if kwarg.arg:
                    self.log_debug(f"kwargs given as keyword parameter not supported: {ast_str(node)}")
                elif self.parent and ast_is_super_call(node):
                    if ast_is_supported_super_call(node, self.self_name, self.log_debug):
                        params = get_mro_parameters(
                            node.func.attr,  # type: ignore
                            get_signature_parameters,
                            self.logger,
                        )
                else:
                    get_param_args = self.get_node_component(node)
                    if get_param_args:
                        params = get_signature_parameters(*get_param_args, logger=self.logger)
                params = remove_given_parameters(node, params)
                if params:
                    self.add_node_origins(params, node)
                    params_list.append(params)
            elif isinstance(node, ast_assign_type):
                self_attr = self.parent and ast_is_attr_assign(node, self.self_name)
                if self_attr:
                    params = self.get_parameters_attr_use_in_members(self_attr)
                    if params:
                        self.add_node_origins(params, node)
                        params_list.append(params)
                else:
                    self.log_debug(f"unsupported type of assign: {ast_str(node)}")

        params = group_parameters(params_list)
        return split_args_and_kwargs(params)

    def get_parameters_attr_use_in_members(self, attr_name) -> ParamList:
        attr_value = ast_attribute_load(self.self_name, attr_name)
        member_names = [
            name
            for name, _ in inspect.getmembers(self.parent)
            if not name.startswith("__") and is_method_or_property(inspect.getattr_static(self.parent, name))
        ]
        for member_name in member_names:
            assert self.parent is not None
            visitor = ParametersVisitor(self.parent, member_name, logger=self.logger)
            kwargs = visitor.get_parameters_call_attr(attr_name, attr_value)
            if kwargs is not None:
                return kwargs
        self.log_debug(f"did not find use of {self.self_name}.{attr_name} in members of {self.parent}")
        return []

    def add_node_origins(self, params: ParamList, node) -> None:
        origin = None
        for param in params:
            if param.origin is None:
                if not origin:
                    origin = f"{get_parameter_origins(self.component, self.parent)}:{node.lineno}"
                param.origin = origin

    def get_parameters_call_attr(self, attr_name: str, attr_value: ast.AST) -> Optional[ParamList]:
        self.parse_source_tree()
        values_to_find = {attr_name: attr_value}
        values_found = self.find_values_usage(values_to_find)
        matched = []
        if values_found:
            for _, node in values_found:
                match = self.match_call_that_uses_attr(node, attr_name)
                if match:
                    self.add_node_origins(match, node)
                    matched.append(match)
            matched = group_parameters(matched)
        return matched or None

    def get_parameters(self) -> ParamList:
        if self.component is None:
            return []
        params, args_idx, kwargs_idx, doc_params, stubs = get_signature_parameters_and_indexes(
            self.component, self.parent, self.logger
        )
        self.replace_param_default_subclass_specs(params)
        if args_idx >= 0 or kwargs_idx >= 0:
            self.doc_params = doc_params
            with mro_context(self.parent):
                args, kwargs = self.get_parameters_args_and_kwargs()
            params = replace_args_and_kwargs(params, args, kwargs)
        add_stub_types(stubs, params, self.component)
        return params


def get_parameters_by_assumptions(
    function_or_class: Union[Callable, Type],
    method_name: Optional[str] = None,
    logger: Union[bool, str, dict, logging.Logger] = True,
) -> ParamList:
    component, parent, method_name = get_component_and_parent(function_or_class, method_name)
    params, args_idx, kwargs_idx, _, stubs = get_signature_parameters_and_indexes(component, parent, logger)

    if parent and (args_idx >= 0 or kwargs_idx >= 0):
        with mro_context(parent):
            subparams = get_mro_parameters(method_name, get_parameters_by_assumptions, logger)
        if subparams:
            args, kwargs = split_args_and_kwargs(subparams)
            params = replace_args_and_kwargs(params, args, kwargs)

    params = replace_args_and_kwargs(params, [], [])
    add_stub_types(stubs, params, component)
    return params


def get_parameters_from_pydantic_or_attrs(
    function_or_class: Union[Callable, Type],
    method_or_property: Optional[str],
    logger: logging.Logger,
) -> Optional[ParamList]:
    from ._optionals import attrs_support, pydantic_support

    if method_or_property or not (pydantic_support or attrs_support):
        return None

    fields_iterator = None
    if pydantic_support:
        from pydantic import BaseModel  # pylint: disable=no-name-in-module

        if is_subclass(function_or_class, BaseModel):
            fields_iterator = function_or_class.__fields__.values()  # type: ignore
            is_required = lambda f: f.required
            is_factory = lambda f: f.default_factory
            run_factory = lambda f: f.default_factory()
            get_annotation = lambda f: f.annotation
            get_doc = lambda f: f.field_info.description or doc_params.get(f.name)  # type: ignore

    if not fields_iterator and attrs_support:
        import attrs

        if attrs.has(function_or_class):  # type: ignore
            fields_iterator = attrs.fields(function_or_class)
            is_required = lambda f: f.default is attrs.NOTHING
            is_factory = lambda f: isinstance(f.default, attrs.Factory)  # type: ignore
            run_factory = lambda f: f.default.factory()
            get_annotation = lambda f: f.type
            get_doc = lambda f: doc_params.get(f.name)  # type: ignore

    if not fields_iterator:
        return None

    params = []
    doc_params = parse_docs(function_or_class, None, logger)
    for field in fields_iterator:
        if is_required(field):
            default = inspect._empty
        elif is_factory(field):
            default = run_factory(field)
        else:
            default = field.default
        params.append(
            ParamData(
                name=field.name,
                annotation=get_annotation(field),
                default=default,
                kind=kinds.KEYWORD_ONLY,
                doc=get_doc(field),
                component=function_or_class,
            )
        )
    evaluate_postponed_annotations(params, function_or_class, logger)
    return params


def get_signature_parameters(
    function_or_class: Union[Callable, Type],
    method_or_property: Optional[str] = None,
    logger: Union[bool, str, dict, logging.Logger] = True,
) -> ParamList:
    """Get parameters by inspecting ASTs or by inheritance assumptions if source not available.

    In contrast to inspect.signature, it follows the use of *args and **kwargs
    attempting to find all accepted named parameters.

    Args:
        function_or_class: The callable object from which to get the signature
            parameters.
        method_or_property: For classes, the name of the method or property from
            which to get the signature parameters. If not provided it returns
            the parameters for ``__init__``.
        logger: Useful for debugging. Only logs at ``DEBUG`` level.
    """
    logger = parse_logger(logger, "get_signature_parameters")
    try:
        params = get_parameters_from_pydantic_or_attrs(function_or_class, method_or_property, logger)
        if params is not None:
            return params
        visitor = ParametersVisitor(function_or_class, method_or_property, logger=logger)
        return visitor.get_parameters()
    except Exception as ex:
        cause = "Source not available"
        exc_info = None
        if not isinstance(ex, SourceNotAvailable):
            cause = "Problems with AST resolving"
            exc_info = ex
        logger.debug(
            f"{cause}, falling back to parameters by assumptions: function_or_class={function_or_class} "
            f"method_or_property={method_or_property}: {ex}",
            exc_info=exc_info,
        )
        return get_parameters_by_assumptions(function_or_class, method_or_property, logger)
