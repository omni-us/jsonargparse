"""Actions to support jsonnet."""

from typing import Any, Dict, Optional, Tuple, Union

from ._actions import Action, _find_action, _is_action_value_list
from ._common import parser_context
from ._jsonschema import ActionJsonSchema
from ._loaders_dumpers import get_loader_exceptions, load_value
from ._optionals import (
    get_config_read_mode,
    get_jsonschema_exceptions,
    import_jsonnet,
    import_jsonschema,
)
from ._typehints import ActionTypeHint
from ._util import Path, argument_error

__all__ = ["ActionJsonnet"]


class ActionJsonnet(Action):
    """Action to parse a jsonnet, optionally validating against a jsonschema."""

    def __init__(
        self,
        ext_vars: Optional[str] = None,
        schema: Optional[Union[str, Dict]] = None,
        **kwargs,
    ):
        """Initializer for ActionJsonnet instance.

        Args:
            ext_vars: Key where to find the external variables required to parse the jsonnet.
            schema: Schema to validate values against.

        Raises:
            ValueError: If a parameter is invalid.
            jsonschema.exceptions.SchemaError: If the schema is invalid.
        """
        if "_validator" not in kwargs:
            import_jsonnet("ActionJsonnet")
            if not isinstance(ext_vars, (str, type(None))):
                raise ValueError("ext_vars has to be either None or a string.")
            self._ext_vars = ext_vars
            if schema is not None:
                jsonvalidator = import_jsonschema("ActionJsonnet")[1]
                if isinstance(schema, str):
                    with parser_context(load_value_mode="yaml"):
                        try:
                            schema = load_value(schema)
                        except get_loader_exceptions() as ex:
                            raise ValueError(f"Problems parsing schema: {ex}") from ex
                jsonvalidator.check_schema(schema)
                self._validator = ActionJsonSchema._extend_jsonvalidator_with_default(jsonvalidator)(schema)
            else:
                self._validator = None
        else:
            self._ext_vars = kwargs.pop("_ext_vars")
            self._validator = kwargs.pop("_validator")
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument as jsonnet using ext_vars if defined.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs["_ext_vars"] = self._ext_vars
            kwargs["_validator"] = self._validator
            if "help" in kwargs and "%s" in kwargs["help"] and self._validator is not None:
                import json

                kwargs["help"] = kwargs["help"] % json.dumps(self._validator.schema, sort_keys=True)
            return ActionJsonnet(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2], cfg=args[1]))
        return None

    def _check_type(self, value, cfg):
        islist = _is_action_value_list(self)
        ext_vars = {}
        if cfg:
            ext_vars = cfg.get(self._ext_vars, {})
        if not islist:
            value = [value]
        for num, val in enumerate(value):
            try:
                if isinstance(val, str):
                    val = self.parse(val, ext_vars=ext_vars, with_meta=True)
                elif self._validator is not None:
                    self._validator.validate(val)
                value[num] = val
            except (TypeError, RuntimeError) + get_jsonschema_exceptions() + get_loader_exceptions() as ex:
                elem = "" if not islist else " element " + str(num + 1)
                raise TypeError(f'Parser key "{self.dest}"{elem}: {ex}') from ex
        return value if islist else value[0]

    @staticmethod
    def _check_ext_vars_action(parser, action):
        if isinstance(action, ActionJsonnet) and action._ext_vars:
            ext_vars_action = _find_action(parser, action._ext_vars)
            if not ext_vars_action:
                raise ValueError(f"No argument found for ext_vars='{action._ext_vars}'")
            ext_vars_type = isinstance(ext_vars_action, ActionTypeHint) and ext_vars_action._typehint
            if ext_vars_type not in {dict, Dict}:
                raise ValueError(
                    f"Type for ext_vars='{action._ext_vars}' argument must be dict, given: {ext_vars_type}"
                )
            if ext_vars_action.default is None:
                ext_vars_action.default = {}
            if not isinstance(ext_vars_action.default, dict):
                raise ValueError(
                    f"Default value for the ext_vars='{action._ext_vars}' argument "
                    f"must be dict or None, given: {ext_vars_action.default}"
                )
            ext_vars_action.jsonnet_ext_vars = True

    @staticmethod
    def split_ext_vars(ext_vars: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Splits an ext_vars dict into the ext_codes and ext_vars required by jsonnet.

        Args:
            ext_vars: External variables. Values can be strings or any other basic type.
        """
        if ext_vars is None:
            ext_vars = {}
        import json

        ext_codes = {k: json.dumps(v) for k, v in ext_vars.items() if not isinstance(v, str)}
        ext_vars = {k: v for k, v in ext_vars.items() if isinstance(v, str)}
        return ext_vars, ext_codes

    def parse(
        self,
        jsonnet: Union[str, Path],
        ext_vars: Optional[Dict[str, Any]] = None,
        with_meta: bool = False,
    ) -> Dict:
        """Method that can be used to parse jsonnet independent from an ArgumentParser.

        Args:
            jsonnet: Either a path to a jsonnet file or the jsonnet content.
            ext_vars: External variables. Values can be strings or any other basic type.
            with_meta: Whether to include metadata in config object.

        Returns:
            The parsed jsonnet object.

        Raises:
            TypeError: If the input is neither a path to an existent file nor a jsonnet.
        """
        _jsonnet = import_jsonnet("ActionJsonnet")
        ext_vars, ext_codes = self.split_ext_vars(ext_vars)
        fpath = None
        fname = "snippet"
        snippet = jsonnet
        try:
            fpath = Path(jsonnet, mode=get_config_read_mode())
        except TypeError:
            pass
        else:
            fname = jsonnet(absolute=False) if isinstance(jsonnet, Path) else jsonnet
            snippet = fpath.get_content()
        try:
            with parser_context(load_value_mode="yaml"):
                values = load_value(_jsonnet.evaluate_snippet(fname, snippet, ext_vars=ext_vars, ext_codes=ext_codes))
        except RuntimeError as ex:
            raise argument_error(f'Problems evaluating jsonnet "{fname}": {ex}') from ex
        if self._validator is not None:
            self._validator.validate(values)
        if with_meta:
            if fpath is not None:
                values["__path__"] = fpath
            values["__orig__"] = snippet
        return values
