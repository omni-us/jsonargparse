import ast
import inspect
import logging
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from .util import ClassFromFunctionBase, is_subclass, LoggerProperty, parse_logger
from .optionals import parse_docs


@dataclass
class ParamData:
    name: str
    annotation: Any
    default: Any = inspect._empty
    kind: Optional[inspect._ParameterKind] = None
    doc: Optional[str] = None
    component: Optional[Union[Callable, Type, Tuple]] = None
    parent: Optional[Type] = None


ParamList = List[ParamData]
parameter_attributes = [s[1:] for s in inspect.Parameter.__slots__]  # type: ignore
kinds = inspect._ParameterKind
ast_assign_type: Tuple[Type[ast.AST], ...] = (ast.AnnAssign, ast.Assign)


class SourceNotAvailable(Exception):
    "Raised when the source code for some component is not available."


def is_staticmethod(attr) -> bool:
    return isinstance(attr, staticmethod)


def is_method(attr) -> bool:
    return inspect.isfunction(attr) and not is_staticmethod(attr)


def is_property(attr) -> bool:
    return isinstance(attr, property)


def is_method_or_property(attr) -> bool:
    return is_method(attr) or is_property(attr)


def is_classmethod(parent, component) -> bool:
    return parent and isinstance(inspect.getattr_static(parent, component.__name__), classmethod)


def ast_str(node):
    return getattr(ast, 'unparse', ast.dump)(node)


def ast_variable_load(name):
    return ast.Name(id=name, ctx=ast.Load())


def ast_attribute_load(container, name):
    return ast.Attribute(value=ast.Name(id=container, ctx=ast.Load()), attr=name, ctx=ast.Load())


def ast_is_assign_with_value(node, value) -> bool:
    return isinstance(node, ast_assign_type) and ast.dump(node.value) == ast.dump(value)


def ast_get_assign_targets(node):
    return node.targets if isinstance(node, ast.Assign) else [node.target]


dict_ast = ast.dump(ast_variable_load('dict'))


def ast_is_dict_assign(node):
    return isinstance(node, ast_assign_type) and isinstance(node.value, ast.Call) and ast.dump(node.value.func) == dict_ast


def ast_is_dict_assign_with_value(node, value):
    if ast_is_dict_assign(node):
        value_dump = ast.dump(value)
        for keyword in [k.value for k in node.value.keywords]:
            if ast.dump(keyword) == value_dump:
                return True
    return False


def ast_is_call_with_value(node, value_dump) -> bool:
    for argtype in ['args', 'keywords']:
        for arg in getattr(node, argtype):
            if isinstance(getattr(arg, 'value', None), ast.AST) and ast.dump(arg.value) == value_dump:
                return True
    return False


ast_constant_attr = {
    ast.Constant: 'value',
    # python <= 3.7:
    ast.NameConstant: 'value',
    ast.Num: 'n',
    ast.Str: 's',
}


def ast_is_constant(node):
    return isinstance(node, (ast.Str, ast.Num, ast.NameConstant, ast.Constant))


def ast_get_constant_value(node):
    assert ast_is_constant(node)
    return getattr(node, ast_constant_attr[node.__class__])


def ast_is_kwargs_pop_or_get(node, value_dump) -> bool:
    return (
        isinstance(node.func, ast.Attribute) and
        value_dump == ast.dump(node.func.value) and
        node.func.attr in {'pop', 'get'} and
        len(node.args) == 2 and
        isinstance(ast_get_constant_value(node.args[0]), str)
    )


def ast_is_super_call(node) -> bool:
    return (
        isinstance(node, ast.Call) and
        isinstance(node.func, ast.Attribute) and
        isinstance(node.func.value, ast.Call) and
        isinstance(node.func.value.func, ast.Name) and
        node.func.value.func.id == 'super'
    )


def ast_is_supported_super_call(node, self_name, logger) -> bool:
    supported = False
    args = node.func.value.args
    if not args and not node.func.value.keywords:
        supported = True
    elif (
        args and
        len(args) == 2 and
        all(isinstance(a, ast.Name) for a in args) and
        self_name == args[1].id and
        not node.func.value.keywords
    ):
        classes, idx = current_mro.get()
        module = inspect.getmodule(classes[idx])
        for offset, cls in enumerate(classes[idx:]):
            if args[0].id == cls.__name__ and cls is getattr(module, cls.__name__, None):
                current_mro.set((classes, idx+offset))
                supported = True
                break
    if not supported:
        logger.debug(f'AST resolver: unsupported super parameters: {ast_str(node)}')
    return supported


def ast_is_attr_assign(node, container):
    for target in ast_get_assign_targets(node) if isinstance(node, ast_assign_type) else []:
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == container:
            return target.attr
    return False


def ast_get_call_kwarg_with_value(node, value):
    value_dump = ast.dump(value)
    for arg in node.keywords:
        if isinstance(getattr(arg, 'value', None), ast.AST) and ast.dump(arg.value) == value_dump:
            return arg


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
    if is_classmethod(parent, component):
        params = list(inspect.signature(component.__func__).parameters.values())
    else:
        params = list(inspect.signature(component).parameters.values())
    if parent:
        params = params[1:]
    args_idx = get_arg_kind_index(params, kinds.VAR_POSITIONAL)
    kwargs_idx = get_arg_kind_index(params, kinds.VAR_KEYWORD)
    doc_params = {}
    for doc in parse_docs(component, parent, logger):
        for param in doc.params:
            doc_params[param.arg_name] = param.description
    for num, param in enumerate(params):
        params[num] = ParamData(
            doc=doc_params.get(param.name),
            parent=parent,
            component=component,
            **{a: getattr(param, a) for a in parameter_attributes},
        )
    return params, args_idx, kwargs_idx, doc_params


ast_literals = {
    ast.dump(ast.parse(v, mode='eval').body): lambda: ast.literal_eval(v)
    for v in ['{}', '[]']
}


def get_kwargs_pop_or_get_parameter(node, component, parent, doc_params, logger):
    name = ast_get_constant_value(node.args[0])
    if ast_is_constant(node.args[1]):
        default = ast_get_constant_value(node.args[1])
    else:
        default = ast.dump(node.args[1])
        if default in ast_literals:
            default = ast_literals[default]()
        else:
            default = None
            logger.debug(f'AST resolver: unsupported kwargs pop/get default: {ast_str(node)}')
    return ParamData(
        name=name,
        annotation=inspect._empty,
        default=default,
        kind=inspect._ParameterKind.KEYWORD_ONLY,
        doc=doc_params.get(name),
        parent=parent,
        component=component,
    )


def split_args_and_kwargs(params: ParamList) -> Tuple[ParamList, ParamList]:
    args = [p for p in params if p.kind is kinds.POSITIONAL_ONLY]
    kwargs = [p for p in params if p.kind in {kinds.KEYWORD_ONLY, kinds.POSITIONAL_OR_KEYWORD}]
    return args, kwargs


def replace_args_and_kwargs(params: ParamList, args: ParamList, kwargs: ParamList) -> ParamList:
    args_idx = get_arg_kind_index(params, kinds.VAR_POSITIONAL)
    kwargs_idx = get_arg_kind_index(params, kinds.VAR_KEYWORD)
    if args_idx >= 0:
        params = params[:args_idx] + args + params[args_idx+1:]
        if kwargs_idx != 1:
            kwargs_idx += len(args) - 1
    if kwargs_idx >= 0:
        existing_names = set(p.name for p in params[:kwargs_idx] + params[kwargs_idx+1:])
        kwargs = [p for p in kwargs if p.name not in existing_names]
        params = params[:kwargs_idx] + kwargs + params[kwargs_idx+1:]
    return params


def common_parameters(params_list: List[ParamList]) -> ParamList:
    if len(params_list) == 1:
        return params_list[0]
    common = []
    params_dict = defaultdict(lambda: [])
    for params in params_list:
        for param in params:
            params_dict[param.name].append(param)
    for params in params_dict.values():
        if (
            len(params) == len(params_list) and
            len(set(p.annotation for p in params)) == 1 and
            kinds.POSITIONAL_ONLY not in set(p.kind for p in params)
        ):
            common.append(params[0])
    return common


def merge_parameters(source: Union[ParamData, ParamList], target: ParamList) -> ParamList:
    if not isinstance(source, list):
        source = [source]
    source_dict = {p.name: p for p in source}
    replace_names = set()
    for p in target:
        if p.name in source_dict and p.annotation is inspect._empty and source_dict[p.name] is not inspect._empty:
            replace_names.add(p.name)
    target = [p for p in target if p.name not in replace_names]
    target_names = set(p.name for p in target)
    return target + [s for s in source if s.name not in target_names]


def has_dunder_new_method(cls, attr_name):
    classes = inspect.getmro(cls)[1:]
    return (
        attr_name == '__init__' and
        cls.__new__ is not object.__new__ and
        not any(cls.__new__ is c.__new__ for c in classes)
    )


current_mro: ContextVar = ContextVar('current_mro', default=(None, None))


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
    for num, cls in enumerate(classes[idx+1:], start=idx+1):
        method = getattr(cls, method_name, None)
        remainder = classes[num+1:] + [object]
        if method and not any(method is getattr(c, method_name, None) for c in remainder):
            current_mro.set((classes, num))
            return get_parameters_fn(cls, method, logger=logger)
    return []


def get_component_and_parent(
    function_or_class: Union[Callable, Type],
    method_or_property: Optional[Union[str, Callable]] = None,
):
    if is_subclass(function_or_class, ClassFromFunctionBase) and method_or_property in {None, '__init__'}:
        function_or_class = function_or_class.wrapped_function  # type: ignore
        method_or_property = None
    elif inspect.isclass(function_or_class) and method_or_property is None:
        method_or_property = '__init__'
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
            component = getattr(function_or_class, '__new__')
        elif is_method(attr):
            component = attr
        elif is_property(attr):
            component = attr.fget
        elif isinstance(attr, classmethod):
            component = getattr(function_or_class, method_or_property)
        elif attr is not object.__init__:
            raise ValueError(f'Invalid or unsupported input: class={function_or_class}, method_or_property={method_or_property}')
    else:
        if not callable(function_or_class):
            raise ValueError(f'Non-callable input: function={function_or_class}')
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
        self.parse_source_tree()

    def parse_source_tree(self):
        """Parses the component's module AST and sets the component and parent nodes."""
        if self.component is None:
            return
        try:
            source = textwrap.dedent(inspect.getsource(self.component))
            tree = ast.parse(source)
            assert isinstance(tree, ast.Module) and len(tree.body) == 1
            self.component_node = tree.body[0]
            self.self_name = self.component_node.args.args[0].arg if self.parent else None
        except Exception as ex:
            raise SourceNotAvailable(f'Problems getting source code for {self.component}: {ex}') from ex

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
            else:
                function_or_class = getattr(module, node.func.id)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if self.parent and ast.dump(node.func.value) == ast.dump(ast_variable_load(self.self_name)):
                function_or_class = self.parent
                method_or_property = node.func.attr
            else:
                container = getattr(module, node.func.value.id)
                if inspect.isclass(container):
                    function_or_class = container
                    method_or_property = node.func.attr
                else:
                    function_or_class = getattr(container, node.func.attr)
        if not function_or_class:
            self.logger.debug(f'AST resolver: not supported: {ast_str(node)}')
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
                    self.logger.debug(f'AST resolver: kwargs attribute given as keyword parameter not supported: {ast_str(node)}')
                else:
                    get_param_args = self.get_node_component(node)
                    if get_param_args:
                        params = get_signature_parameters(*get_param_args, logger=self.logger)
            params = remove_given_parameters(node, params)
        return params

    def get_parameters_args_and_kwargs(self) -> Tuple[ParamList, ParamList]:
        args_name = getattr(self.component_node.args.vararg, 'arg', None)
        kwargs_name = getattr(self.component_node.args.kwarg, 'arg', None)
        values_to_find = {}
        if args_name:
            values_to_find[args_name] = ast_variable_load(args_name)
        if kwargs_name:
            values_to_find[kwargs_name] = kwargs_value = ast_variable_load(kwargs_name)

        args: ParamList = []
        kwargs: ParamList = []
        values_found = self.find_values_usage(values_to_find)
        if not values_found:
            return args, kwargs

        kwargs_value_dump = ast.dump(kwargs_value)
        for node in [v for k, v in values_found if k == kwargs_name]:
            if isinstance(node, ast.Call):
                if ast_is_kwargs_pop_or_get(node, kwargs_value_dump):
                    param = get_kwargs_pop_or_get_parameter(node, self.component, self.parent, self.doc_params, self.logger)
                    kwargs = merge_parameters(param, kwargs)
                    continue
                kwarg = ast_get_call_kwarg_with_value(node, kwargs_value)
                params = []
                if kwarg.arg:
                    self.logger.debug(f'AST resolver: kwargs given as keyword parameter not supported: {ast_str(node)}')
                elif self.parent and ast_is_super_call(node):
                    if ast_is_supported_super_call(node, self.self_name, self.logger):
                        params = get_mro_parameters(node.func.attr, get_signature_parameters, self.logger)  # type: ignore
                else:
                    get_param_args = self.get_node_component(node)
                    if get_param_args:
                        params = get_signature_parameters(*get_param_args, logger=self.logger)
                args, kwargs_ = split_args_and_kwargs(remove_given_parameters(node, params))
                kwargs = merge_parameters(kwargs_, kwargs)
                break
            elif isinstance(node, ast_assign_type):
                self_attr = self.parent and ast_is_attr_assign(node, self.self_name)
                if self_attr:
                    params = self.get_parameters_attr_use_in_members(self_attr)
                    kwargs = merge_parameters(params, kwargs)
                    break
                else:
                    self.logger.debug(f'AST resolver: unsupported type of assign: {ast_str(node)}')

        return args, kwargs

    def get_parameters_attr_use_in_members(self, attr_name) -> ParamList:
        attr_value = ast_attribute_load(self.self_name, attr_name)
        member_names = [
            name for name, _ in inspect.getmembers(self.parent)
            if not name.startswith('__') and is_method_or_property(inspect.getattr_static(self.parent, name))
        ]
        for member_name in member_names:
            assert self.parent is not None
            visitor = ParametersVisitor(self.parent, member_name, logger=self.logger)
            kwargs = visitor.get_parameters_call_attr(attr_name, attr_value)
            if kwargs is not None:
                return kwargs
        self.logger.debug(f'AST resolver: did not find use of {self.self_name}.{attr_name} in members of {self.parent}')
        return []

    def get_parameters_call_attr(self, attr_name: str, attr_value: ast.AST) -> Optional[ParamList]:
        values_to_find = {attr_name: attr_value}
        values_found = self.find_values_usage(values_to_find)
        matched = []
        if values_found:
            for _, node in values_found:
                match = self.match_call_that_uses_attr(node, attr_name)
                if match is not None:
                    matched.append(match)
            matched = common_parameters(matched)
        return matched or None

    def get_parameters(self) -> ParamList:
        if self.component is None:
            return []
        params, args_idx, kwargs_idx, doc_params = get_signature_parameters_and_indexes(self.component, self.parent, self.logger)
        if args_idx >= 0 or kwargs_idx >= 0:
            self.doc_params = doc_params
            with mro_context(self.parent):
                args, kwargs = self.get_parameters_args_and_kwargs()
            params = replace_args_and_kwargs(params, args, kwargs)
        return params


def get_parameters_by_assumptions(
    function_or_class: Union[Callable, Type],
    method_name: Optional[str] = None,
    logger: Union[bool, str, dict, logging.Logger] = True,
) -> ParamList:
    component, parent, method_name = get_component_and_parent(function_or_class, method_name)
    params, args_idx, kwargs_idx, _ = get_signature_parameters_and_indexes(component, parent, logger)

    if parent and (args_idx >= 0 or kwargs_idx >= 0):
        with mro_context(parent):
            subparams = get_mro_parameters(method_name, get_parameters_by_assumptions, logger)
        if subparams:
            args, kwargs = split_args_and_kwargs(subparams)
            params = replace_args_and_kwargs(params, args, kwargs)

    params = replace_args_and_kwargs(params, [], [])
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
    logger = parse_logger(logger, 'get_signature_parameters')
    try:
        visitor = ParametersVisitor(function_or_class, method_or_property, logger=logger)
        return visitor.get_parameters()
    except Exception as ex:
        cause = 'Source not available' if isinstance(ex, SourceNotAvailable) else 'Problems with AST resolving'
        logger.debug(
            f'{cause}, falling back to parameters by assumptions: function_or_class={function_or_class} '
            f'method_or_property={method_or_property} :: {ex}'
        )
        return get_parameters_by_assumptions(function_or_class, method_or_property, logger)
