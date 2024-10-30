import argparse
import inspect
import locale
import os
import re
from collections import defaultdict
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from enum import Enum
from importlib.util import find_spec
from subprocess import PIPE, Popen
from typing import List, Union

from ._actions import ActionConfigFile, _ActionConfigLoad, _ActionHelpClassPath, remove_actions
from ._parameter_resolvers import get_signature_parameters
from ._typehints import (
    ActionTypeHint,
    callable_origin_types,
    get_all_subclass_paths,
    get_callable_return_type,
    get_typehint_origin,
    is_subclass,
    type_to_str,
)
from ._util import NoneType, Path, import_object, unique


def handle_completions(parser):
    if find_spec("argcomplete") and "_ARGCOMPLETE" in os.environ:
        import argcomplete

        from ._common import parser_context

        with parser_context(load_value_mode=parser.parser_mode):
            argcomplete.autocomplete(parser)

    if find_spec("shtab") and not getattr(parser, "parent_parser", None):
        if not any(isinstance(action, ShtabAction) for action in parser._actions):
            parser.add_argument("--print_shtab", action=ShtabAction)


# argcomplete


def get_files_completer():
    from argcomplete.completers import FilesCompleter

    return FilesCompleter()


def argcomplete_namespace(caller, parser, namespace):
    if caller == "argcomplete":
        namespace.__class__ = __import__("jsonargparse").Namespace
        namespace = parser.merge_config(parser.get_defaults(skip_check=True), namespace).as_flat()
    return namespace


def argcomplete_warn_redraw_prompt(prefix, message):
    import argcomplete

    if prefix != "":
        argcomplete.warn(message)
        with suppress(Exception):
            proc = Popen(f"ps -p {os.getppid()} -oppid=".split(), stdout=PIPE, stderr=PIPE)
            stdout, _ = proc.communicate()
            shell_pid = int(stdout.decode().strip())
            os.kill(shell_pid, 28)
    _ = "_" if locale.getlocale()[1] != "UTF-8" else "\xa0"
    return [_ + message.replace(" ", _), ""]


# shtab

shtab_shell: ContextVar = ContextVar("shtab_shell")
shtab_prog: ContextVar = ContextVar("shtab_prog")
shtab_preambles: ContextVar = ContextVar("shtab_preambles")


class ShtabAction(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
    ):
        import shtab

        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            choices=shtab.SUPPORTED_SHELLS,
            help="Print shtab shell completion script.",
        )

    def __call__(self, parser, namespace, shell, option_string=None):
        import shtab

        prog = norm_name(parser.prog)
        assert prog
        preambles = []
        if shell == "bash":
            preambles = [bash_compgen_typehint.strip().replace("%s", prog)]
        with prepare_actions_context(shell, prog, preambles):
            shtab_prepare_actions(parser)
        print(shtab.complete(parser, shell, preamble="\n".join(preambles)))
        parser.exit(0)


@contextmanager
def prepare_actions_context(shell, prog, preambles):
    token_shell = shtab_shell.set(shell)
    token_prog = shtab_prog.set(prog)
    token_preambles = shtab_preambles.set(preambles)
    try:
        yield
    finally:
        shtab_shell.reset(token_shell)
        shtab_prog.reset(token_prog)
        shtab_preambles.reset(token_preambles)


def norm_name(name: str) -> str:
    return re.sub(r"\W+", "_", name)


def shtab_prepare_actions(parser) -> None:
    remove_actions(parser, (ShtabAction,))
    if parser._subcommands_action:
        for subparser in parser._subcommands_action._name_parser_map.values():
            shtab_prepare_actions(subparser)
    for action in parser._actions:
        shtab_prepare_action(action, parser)


def shtab_prepare_action(action, parser) -> None:
    import shtab

    if action.choices or hasattr(action, "complete"):
        return

    complete = None
    if isinstance(action, (ActionConfigFile, _ActionConfigLoad)):
        complete = shtab.FILE
    elif isinstance(action, ActionTypeHint):
        typehint = action._typehint
        if get_typehint_origin(typehint) == Union:
            subtypes = [s for s in typehint.__args__ if s not in {NoneType, str, dict, list, tuple, bytes}]
            if len(subtypes) == 1:
                typehint = subtypes[0]
        if is_subclass(typehint, Path):
            if "f" in typehint._mode:
                complete = shtab.FILE
            elif "d" in typehint._mode:
                complete = shtab.DIRECTORY
        elif is_subclass(typehint, os.PathLike):
            complete = shtab.FILE
    if complete:
        action.complete = complete
        return

    choices = None
    if isinstance(action, ActionTypeHint):
        skip = getattr(action, "sub_add_kwargs", {}).get("skip", set())
        choices = get_typehint_choices(action._typehint, action.option_strings[0], parser, skip)
        if shtab_shell.get() == "bash":
            message = f"Expected type: {type_to_str(action._typehint)}"
            add_bash_typehint_completion(parser, action, message, choices)
            choices = None
    elif isinstance(action, _ActionHelpClassPath):
        choices = get_help_class_choices(action._typehint)
    if choices:
        action.choices = choices


bash_compgen_typehint_name = "_jsonargparse_%s_compgen_typehint"
bash_compgen_typehint = """
_jsonargparse_%%s_matched_choices() {
  local TOTAL=$(echo "$1" | wc -w | tr -d " ")
  if [ "$TOTAL" != 0 ]; then
    local MATCH=$(echo "$2" | wc -w | tr -d " ")
    printf "; $MATCH/$TOTAL matched choices"
  fi
}
%(name)s() {
  local MATCH=( $(IFS=" " compgen -W "$1" "$2") )
  if [ ${#MATCH[@]} = 0 ]; then
    if [ "$COMP_TYPE" = 63 ]; then
      MATCHED=$(_jsonargparse_%%s_matched_choices "$1" "${MATCH[*]}")
      printf "%(b)s\\n$3$MATCHED\\n%(n)s" >&2
      kill -WINCH $$
    fi
  else
    IFS=" " compgen -W "$1" "$2"
    if [ "$COMP_TYPE" = 63 ]; then
      MATCHED=$(_jsonargparse_%%s_matched_choices "$1" "${MATCH[*]}")
      printf "%(b)s\\n$3$MATCHED%(n)s" >&2
    fi
  fi
}
""" % {
    "name": bash_compgen_typehint_name,
    "b": "$(tput setaf 5)",
    "n": "$(tput sgr0)",
}


def add_bash_typehint_completion(parser, action, message, choices) -> None:
    fn_typehint = norm_name(bash_compgen_typehint_name % shtab_prog.get())
    fn_name = parser.prog.replace(" [options] ", "_")
    fn_name = norm_name(f"_jsonargparse_{fn_name}_{action.dest}_typehint")
    fn = '{fn_name}(){{ {fn_typehint} "{choices}" "$1" "{message}"; }}'.format(
        fn_name=fn_name,
        fn_typehint=fn_typehint,
        choices=" ".join(choices),
        message=message,
    )
    shtab_preambles.get().append(fn)
    action.complete = {"bash": fn_name}


def get_typehint_choices(typehint, prefix, parser, skip, choices=None, added_subclasses=None) -> List[str]:
    if choices is None:
        choices = []
    if not added_subclasses:
        added_subclasses = set()
    if typehint is bool:
        choices.extend(["true", "false"])
    elif typehint is NoneType:
        choices.append("null")
    elif is_subclass(typehint, Enum):
        choices.extend(list(typehint.__members__))
    else:
        origin = get_typehint_origin(typehint)
        if origin == Union:
            for subtype in typehint.__args__:
                if subtype in added_subclasses or subtype is object:
                    continue
                get_typehint_choices(subtype, prefix, parser, skip, choices, added_subclasses)
        elif ActionTypeHint.is_subclass_typehint(typehint):
            added_subclasses.add(typehint)
            choices.extend(add_subactions_and_get_subclass_choices(typehint, prefix, parser, skip, added_subclasses))
        elif origin in callable_origin_types:
            return_type = get_callable_return_type(typehint)
            if return_type and ActionTypeHint.is_subclass_typehint(return_type):
                num_args = len(typehint.__args__) - 1
                skip.add(num_args)
                choices.extend(
                    add_subactions_and_get_subclass_choices(return_type, prefix, parser, skip, added_subclasses)
                )

    return [] if choices == ["null"] else choices


def add_subactions_and_get_subclass_choices(typehint, prefix, parser, skip, added_subclasses) -> List[str]:
    choices = []
    paths = get_all_subclass_paths(typehint)
    init_args = defaultdict(list)
    subclasses = defaultdict(list)
    for path in paths:
        choices.append(path)
        try:
            cls = import_object(path)
            params = get_signature_parameters(cls, None, parser._logger)
        except Exception as ex:
            parser._logger.debug(f"Unable to get signature parameters for '{path}': {ex}")
            continue
        num_skip = next((s for s in skip if isinstance(s, int)), 0)
        if num_skip > 0:
            params = params[num_skip:]
        for param in params:
            if param.name not in skip:
                init_args[param.name].append(param.annotation)
                subclasses[param.name].append(path.rsplit(".", 1)[-1])

    for name, subtypes in init_args.items():
        option_string = f"{prefix}.{name}"
        if option_string not in parser._option_string_actions:
            action = parser.add_argument(option_string)
            for subtype in unique(subtypes):
                subchoices = get_typehint_choices(subtype, option_string, parser, skip, None, added_subclasses)
                if shtab_shell.get() == "bash":
                    message = f"Expected type: {type_to_str(subtype)}; "
                    message += f"Accepted by subclasses: {', '.join(subclasses[name])}"
                    add_bash_typehint_completion(parser, action, message, subchoices)
                elif subchoices:
                    action.choices = subchoices

    return choices


def get_help_class_choices(typehint) -> List[str]:
    choices = []
    if get_typehint_origin(typehint) == Union:
        for subtype in typehint.__args__:
            if inspect.isclass(subtype):
                choices.extend(get_help_class_choices(subtype))
    else:
        choices = get_all_subclass_paths(typehint)
    return choices
