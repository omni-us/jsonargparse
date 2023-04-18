import ast
from collections import namedtuple
from copy import deepcopy
from typing import Dict, FrozenSet, List, Set, Tuple, Type, Union

var_map = namedtuple('var_map', 'name value')
none_map = var_map(name='NoneType', value=type(None))
union_map = var_map(name='Union', value=Union)
pep585_map = {
    'dict': var_map(name='Dict', value=Dict),
    'frozenset': var_map(name='FrozenSet', value=FrozenSet),
    'list': var_map(name='List', value=List),
    'set': var_map(name='Set', value=Set),
    'tuple': var_map(name='Tuple', value=Tuple),
    'type': var_map(name='Type', value=Type),
}


class BackportTypeHints(ast.NodeTransformer):

    _typing = __import__('typing')

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
        name = f'_{self.__class__.__name__}_{var.name}'
        self.exec_vars[name] = var.value
        return ast.Name(id=name, ctx=ast.Load())

    def backport(self, input_ast: ast.AST, exec_vars: dict) -> ast.AST:
        for key, value in exec_vars.items():
            if getattr(value, '__module__', '') == 'collections.abc':
                if hasattr(self._typing, key):
                    exec_vars[key] = getattr(self._typing, key)
        self.exec_vars = exec_vars
        backport_ast = self.visit(deepcopy(input_ast))
        return ast.fix_missing_locations(backport_ast)
