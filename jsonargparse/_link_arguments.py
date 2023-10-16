"""Code related to argument linking."""

import re
from argparse import SUPPRESS
from argparse import Action as ArgparseAction
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from ._actions import (
    Action,
    ActionConfigFile,
    _ActionConfigLoad,
    _ActionPrintConfig,
    _ActionSubCommands,
    _find_parent_action,
    filter_default_actions,
)
from ._namespace import Namespace, split_key_leaf
from ._parameter_resolvers import get_signature_parameters
from ._type_checking import ArgumentParser, _ArgumentGroup

__all__ = ["ArgumentLinking"]


def find_parent_or_child_actions(
    parser: "ArgumentParser",
    key: str,
    exclude: Optional[Union[Type[ArgparseAction], Tuple[Type[ArgparseAction], ...]]] = None,
) -> Optional[List[ArgparseAction]]:
    found: List[ArgparseAction] = []
    action = _find_parent_action(parser, key, exclude=exclude)
    if action is not None:
        found = [action]
    else:
        actions = filter_default_actions(parser._actions)
        if exclude is not None:
            actions = [a for a in actions if not isinstance(a, exclude)]
        prefix = key + "."
        found = [a for a in actions if a.dest.startswith(prefix)]
    return None if found == [] else found


def find_subclass_action_or_class_group(
    parser: "ArgumentParser",
    key: str,
    exclude: Optional[Union[Type[ArgparseAction], Tuple[Type[ArgparseAction], ...]]] = None,
) -> Optional[Union[ArgparseAction, "_ArgumentGroup"]]:
    from ._typehints import ActionTypeHint

    action = _find_parent_action(parser, key, exclude=exclude)
    if ActionTypeHint.is_subclass_typehint(action):
        return action
    key_set = {key, split_key_leaf(key)[0]}
    for group in parser._action_groups:
        if getattr(group, "dest", None) in key_set and hasattr(group, "instantiate_class"):
            return group
    return None


apply_config_skip: ContextVar = ContextVar("apply_config_skip", default=False)


@contextmanager
def skip_apply_links():
    t = apply_config_skip.set(True)
    try:
        yield
    finally:
        apply_config_skip.reset(t)


class DirectedGraph:
    def __init__(self):
        self.nodes = []
        self.edges_dict = defaultdict(list)

    def add_edge(self, source, target):
        for node in [source, target]:
            if node not in self.nodes:
                self.nodes.append(node)
        self.edges_dict[self.nodes.index(source)].append(self.nodes.index(target))

    def get_topological_order(self):
        exploring = [False] * len(self.nodes)
        visited = [False] * len(self.nodes)
        order = []
        for source in range(len(self.nodes)):
            if not visited[source]:
                self.topological_sort(source, exploring, visited, order)
        return [self.nodes[n] for n in order]

    def topological_sort(self, source, exploring, visited, order):
        exploring[source] = True
        for target in self.edges_dict[source]:
            if exploring[target]:
                raise ValueError(
                    f"Graph has cycles, found while checking {self.nodes[source]} --> " + self.nodes[target]
                )
            elif not visited[target]:
                self.topological_sort(target, exploring, visited, order)
        visited[source] = True
        exploring[source] = False
        order.insert(0, source)


class ActionLink(Action):
    def __init__(
        self,
        parser,
        source: Union[str, Tuple[str, ...]],
        target: str,
        compute_fn: Optional[Callable] = None,
        apply_on: str = "parse",
    ):
        if not hasattr(parser, "_links_group"):
            parser._links_group = parser.add_argument_group("Linked arguments")
        self.parser = parser
        self.apply_on = apply_on
        self.compute_fn = compute_fn
        self._initial_input_checks(source, target)

        # Set and check source actions or group
        exclude = (ActionLink, _ActionConfigLoad, _ActionSubCommands, ActionConfigFile)
        source = (source,) if isinstance(source, str) else source
        if apply_on == "instantiate":
            self.source = [(s, find_subclass_action_or_class_group(parser, s, exclude=exclude)) for s in source]
            for key, action in self.source:
                if action is None:
                    raise ValueError(
                        f"Links applied on instantiation require source to be a subclass action or a class group: {key}"
                    )
        else:
            self.source = [
                (s, find_parent_or_child_actions(parser, s, exclude=exclude)) for s in source  # type: ignore
            ]

        # Set and check target action
        self.target = (target, _find_parent_action(parser, target, exclude=exclude))
        for key, action in self.source + [self.target]:
            if action is None:
                raise ValueError(f'No action for key "{key}".')
        assert self.target[1] is not None

        from ._typehints import ActionTypeHint

        is_target_subclass = ActionTypeHint.is_subclass_typehint(self.target[1], all_subtypes=False, also_lists=True)
        valid_target_init_arg = is_target_subclass and target.startswith(f"{self.target[1].dest}.init_args.")
        valid_target_leaf = self.target[1].dest == target
        if not valid_target_leaf and is_target_subclass and not valid_target_init_arg:
            prefix = f"{self.target[1].dest}.init_args."
            raise ValueError(f'Target key expected to start with "{prefix}", got "{target}".')

        # Replace target action with link action
        if not is_target_subclass or valid_target_leaf:
            for key in self.target[1].option_strings:
                parser._option_string_actions[key] = self
            parser._actions[parser._actions.index(self.target[1])] = self
            for group in parser._action_groups:
                if self.target[1] in group._group_actions:
                    group._group_actions.remove(self.target[1])
                    if is_target_subclass:
                        help_dest = f"{self.target[1].dest}.help"
                        for action in group._group_actions:
                            if action.dest == help_dest:  # type: ignore
                                group._group_actions.remove(action)
                                break
                    if group._group_actions and all(isinstance(a, _ActionConfigLoad) for a in group._group_actions):
                        group.description = (
                            f"Group '{group._group_actions[0].dest}': All arguments are derived from links."
                        )
                        group._group_actions.clear()

        # Remove target from required
        if target in parser.required_args:
            parser.required_args.remove(target)
        if is_target_subclass and not valid_target_leaf:
            sub_add_kwargs = getattr(self.target[1], "sub_add_kwargs", {})
            if "linked_targets" not in sub_add_kwargs:
                sub_add_kwargs["linked_targets"] = set()
            subtarget = target.split(".init_args.", 1)[1]
            sub_add_kwargs["linked_targets"].add(subtarget)

        # Add link action to group to show in help
        parser._links_group._group_actions.append(self)

        # Check instantiation link does not create cycle
        if apply_on == "instantiate":
            try:
                self.instantiation_order(parser)
            except ValueError as ex:
                raise ValueError(f"Invalid link {source[0]} --> {target}: {ex}") from ex

        # Initialize link action
        if compute_fn is None:
            link_str = source[0]
        else:
            link_str = getattr(compute_fn, "__name__", str(compute_fn)) + "(" + ", ".join(source) + ")"
        link_str += " --> " + target

        help_str: Optional[str]
        if is_target_subclass and not valid_target_leaf:
            type_attr = None
            help_str = f"Use --{self.target[1].dest}.help CLASS_PATH for details."
        else:
            type_attr = getattr(self.target[1], "_typehint", self.target[1].type)
            help_str = self.target[1].help
            if help_str == import_module("jsonargparse._formatters").empty_help:
                help_str = f"Target argument '{self.target[1].dest}' lacks type and help"

        super().__init__(
            [link_str],
            dest=target,
            default=SUPPRESS,
            metavar=f"[applied on {self.apply_on}]",
            type=type_attr,
            help=help_str,
        )

    def _initial_input_checks(self, source, target):
        # Check apply_on
        if self.apply_on not in {"parse", "instantiate"}:
            raise ValueError("apply_on must be 'parse' or 'instantiate'.")

        # Check compute function
        if self.compute_fn is None and not isinstance(source, str):
            raise ValueError("Multiple source keys requires a compute function.")

        if self.apply_on == "parse":
            # Check source
            link_actions = self.parser._links_group._group_actions
            existing_targets = {a.target[0] for a in link_actions}
            for src in [source] if isinstance(source, str) else source:
                if src in existing_targets:
                    raise ValueError(f'Source "{src}" not allowed since it is the target of another link.')
            # Check target
            existing_sources = {s[0] for a in link_actions for s in a.source if a.apply_on == "parse"}
            if target in existing_sources:
                raise ValueError(f'Target "{target}" not allowed since it is the source of another link.')

    def __call__(self, *args, **kwargs):
        source = ", ".join(s[0] for s in self.source)
        raise TypeError(f'Linked "{self.target[0]}" must be given via "{source}".')

    def _check_type(self, value, cfg=None):
        return self.parser._check_value_key(self.target[1], value, self.target[0], cfg)

    def call_compute_fn(self, args):
        try:
            assert callable(self.compute_fn)
            return self.compute_fn(*args)
        except Exception as ex:
            link = self.option_strings[0]
            args = ", ".join(str(a) for a in args)
            raise ValueError(f"Call to compute_fn of link '{link}' with args ({args}) failed: {ex}") from ex

    @staticmethod
    def apply_parsing_links(parser: "ArgumentParser", cfg: Namespace) -> None:
        if apply_config_skip.get() or _ActionPrintConfig.is_print_config_requested(parser):
            return

        subcommand, subparser = _ActionSubCommands.get_subcommand(parser, cfg, fail_no_subcommand=False)
        if subcommand and subcommand in cfg:
            ActionLink.apply_parsing_links(subparser, cfg[subcommand])  # type: ignore
        if not hasattr(parser, "_links_group"):
            return
        for action in parser._links_group._group_actions:
            if action.apply_on != "parse":
                continue
            from ._typehints import ActionTypeHint

            args = []
            skip_link = False
            for source_key, source_action in action.source:
                if ActionTypeHint.is_subclass_typehint(source_action[0]) and source_key not in cfg:
                    parser.logger.debug(
                        f"Link '{action.option_strings[0]}' ignored since source '{source_key}' not found in namespace."
                    )
                    skip_link = True
                    break
                for source_action_n in [a for a in source_action if a.dest in cfg]:
                    parser._check_value_key(source_action_n, cfg[source_action_n.dest], source_action_n.dest, None)
                args.append(cfg[source_key])
            if skip_link:
                continue

            if action.compute_fn is None:
                value = args[0]
                # Automatic namespace to dict based on link target type hint
                target_key, target_action = action.target
                if isinstance(value, Namespace) and isinstance(target_action, ActionTypeHint):
                    same_key = target_key == target_action.dest
                    if (
                        same_key and target_action.is_mapping_typehint(target_action._typehint)
                    ) or target_action.is_init_arg_mapping_typehint(target_key, cfg):
                        value = value.as_dict()
            else:
                # Automatic namespace to dict based on compute_fn param type hint
                params = get_signature_parameters(action.compute_fn)
                for n, param in enumerate(params):
                    if (
                        n < len(args)
                        and isinstance(args[n], Namespace)
                        and ActionTypeHint.is_mapping_typehint(param.annotation)
                    ):
                        args[n] = args[n].as_dict()
                # Compute value
                value = action.call_compute_fn(args)
            ActionLink.set_target_value(action, value, cfg, parser.logger)

    @staticmethod
    def apply_instantiation_links(parser, cfg, target=None, order=None):
        if not hasattr(parser, "_links_group"):
            return

        applied_key = "__applied_instantiation_links__"
        applied_links = cfg.pop(applied_key) if applied_key in cfg else set()
        link_actions = [
            a for a in parser._links_group._group_actions if a.apply_on == "instantiate" and a not in applied_links
        ]
        if order and link_actions:
            link_actions = ActionLink.reorder(order, link_actions)

        for action in link_actions:
            if not (order or action.target[0] == target or action.target[0].startswith(target + ".")):
                continue
            source_objects = []
            for source_key, source_action in action.source:
                source_object = cfg[source_action.dest]
                if source_key == source_action.dest:
                    source_objects.append(source_object)
                else:
                    attr = split_key_leaf(source_key)[1]
                    from ._typehints import ActionTypeHint

                    if ActionTypeHint.is_subclass_typehint(source_action) and not hasattr(source_object, attr):
                        parser.logger.debug(
                            f"Link '{action.option_strings[0]}' ignored since attribute '{attr}' not found "
                            f"in source {source_object}."
                        )
                        continue
                    source_objects.append(getattr(source_object, attr))
            if not source_objects:
                continue
            elif action.compute_fn is None:
                value = source_objects[0]
            else:
                value = action.call_compute_fn(source_objects)
            ActionLink.set_target_value(action, value, cfg, parser.logger)
            applied_links.add(action)

        if target:
            cfg[applied_key] = applied_links

    @staticmethod
    def set_target_value(action: "ActionLink", value: Any, cfg: Namespace, logger) -> None:
        target_key, target_action = action.target
        assert target_action
        from ._typehints import ActionTypeHint

        if ActionTypeHint.is_subclass_typehint(target_action, all_subtypes=False, also_lists=True):
            if target_key == target_action.dest:
                target_action._check_type(value)  # type: ignore
            else:
                parent = cfg.get(target_action.dest)
                child_key = target_key[len(target_action.dest) + 1 :]
                if isinstance(parent, list) and any(isinstance(i, Namespace) and child_key in i for i in parent):
                    for item in parent:
                        if child_key in item:
                            item[child_key] = value
                    return
                if target_key not in cfg:
                    logger.debug(f"Link '{action.option_strings[0]}' ignored since target not found.")
                    return
        cfg[target_key] = value

    @staticmethod
    def instantiation_order(parser):
        if hasattr(parser, "_links_group"):
            actions = [a for a in parser._links_group._group_actions if a.apply_on == "instantiate"]
            if len(actions) > 0:
                graph = DirectedGraph()
                for action in actions:
                    target = re.sub(r"\.init_args$", "", split_key_leaf(action.target[0])[0])
                    for _, source_action in action.source:
                        graph.add_edge(source_action.dest, target)
                return graph.get_topological_order()
        return []

    @staticmethod
    def reorder(order, components):
        ordered = []
        for key in order:
            after = []
            for component in components:
                if key == component.dest or component.dest.startswith(key + "."):
                    ordered.append(component)
                else:
                    after.append(component)
            components = after
        return ordered + components

    @staticmethod
    def strip_link_target_keys(parser, cfg):
        def del_target_key(target_key):
            cfg.pop(target_key, None)
            if "." not in target_key:
                return
            parent_key, _ = split_key_leaf(target_key)
            if "." in target_key and parent_key in cfg and not cfg[parent_key]:
                del cfg[parent_key]

        for action in [a for a in parser._actions if isinstance(a, ActionLink)]:
            del_target_key(action.target[0])
        from ._typehints import ActionTypeHint

        for action in [a for a in parser._actions if isinstance(a, ActionTypeHint) and hasattr(a, "sub_add_kwargs")]:
            for key in action.sub_add_kwargs.get("linked_targets", []):
                del_target_key(f"{action.dest}.init_args.{key}")

        with _ActionSubCommands.not_single_subcommand():
            subcommands, subparsers = _ActionSubCommands.get_subcommands(parser, cfg)
        if subcommands is not None:
            for num, subcommand in enumerate(subcommands):
                if subcommand in cfg:
                    ActionLink.strip_link_target_keys(subparsers[num], cfg[subcommand])


class ArgumentLinking:
    """Method for linking arguments."""

    def link_arguments(
        self,
        source: Union[str, Tuple[str, ...]],
        target: str,
        compute_fn: Optional[Callable] = None,
        apply_on: str = "parse",
    ):
        """Makes an argument value be derived from the values of other arguments.

        Refer to :ref:`argument-linking` for a detailed explanation and examples.

        Args:
            source: Key(s) from which the target value is derived.
            target: Key to where the value is set.
            compute_fn: Function to compute target value from source.
            apply_on: At what point to set target value, 'parse' or 'instantiate'.

        Raises:
            ValueError: If an invalid parameter is given.
        """
        ActionLink(self, source, target, compute_fn, apply_on)
