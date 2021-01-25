"""Actions to support jsonnet."""

import json
import yaml
from typing import Optional, Union, Dict, Tuple, Any
from argparse import Action, Namespace

from .actions import _is_action_value_list
from .jsonschema import ActionJsonSchema
from .optionals import (
    import_jsonschema,
    import_jsonnet,
    get_config_read_mode,
    jsonschemaValidationError,
)
from .util import (
    dict_to_namespace,
    namespace_to_dict,
    yamlParserError,
    yamlScannerError,
    ParserError,
    Path,
    _get_key_value,
    _flat_namespace_to_dict,
    _dict_to_flat_namespace,
)


__all__ = [
    'ActionJsonnetExtVars',
    'ActionJsonnet',
]


class ActionJsonnetExtVars(ActionJsonSchema):
    """Action to add argument to provide ext_vars for jsonnet parsing."""

    def __init__(self, **kwargs):
        """Initializer for ActionJsonnetExtVars instance."""
        super().__init__(schema={'type': 'object'}, with_meta=False)


    def __call__(self, *args, **kwargs):
        if len(args) == 0 and 'default' not in kwargs:
            kwargs['default'] = {}
        return super().__call__(*args, **kwargs)


class ActionJsonnet(Action):
    """Action to parse a jsonnet, optionally validating against a jsonschema."""

    def __init__(
        self,
        ext_vars: str = None,
        schema: Union[str, Dict] = None,
        **kwargs
    ):
        """Initializer for ActionJsonnet instance.

        Args:
            ext_vars: Key where to find the external variables required to parse the jsonnet.
            schema: Schema to validate values against.

        Raises:
            ValueError: If a parameter is invalid.
            jsonschema.exceptions.SchemaError: If the schema is invalid.
        """
        if '_validator' not in kwargs:
            import_jsonnet('ActionJsonnet')
            if not isinstance(ext_vars, (str, type(None))):
                raise ValueError('ext_vars has to be either None or a string.')
            self._ext_vars = ext_vars
            if schema is not None:
                jsonvalidator = import_jsonschema('ActionJsonnet')[1]
                if isinstance(schema, str):
                    try:
                        schema = yaml.safe_load(schema)
                    except (yamlParserError, yamlScannerError) as ex:
                        raise ValueError('Problems parsing schema :: '+str(ex)) from ex
                jsonvalidator.check_schema(schema)
                self._validator = ActionJsonSchema._extend_jsonvalidator_with_default(jsonvalidator)(schema)
            else:
                self._validator = None
        else:
            self._ext_vars = kwargs.pop('_ext_vars')
            self._validator = kwargs.pop('_validator')
            super().__init__(**kwargs)


    def __call__(self, *args, **kwargs):
        """Parses an argument as jsonnet using ext_vars if defined.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs['_ext_vars'] = self._ext_vars
            kwargs['_validator'] = self._validator
            if 'help' in kwargs and '%s' in kwargs['help'] and self._validator is not None:
                kwargs['help'] = kwargs['help'] % json.dumps(self._validator.schema, sort_keys=True)
            return ActionJsonnet(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2], cfg=args[1]))


    def _check_type(self, value, cfg):
        islist = _is_action_value_list(self)
        ext_vars = {}
        if self._ext_vars is not None:
            try:
                if isinstance(cfg, dict):
                    cfg = _flat_namespace_to_dict(_dict_to_flat_namespace(cfg))
                ext_vars = _get_key_value(cfg, self._ext_vars)
            except (KeyError, AttributeError):
                pass
        if not islist:
            value = [value]
        for num, val in enumerate(value):
            try:
                if isinstance(val, str):
                    val = self.parse(val, ext_vars=ext_vars, with_meta=True)
                elif self._validator is not None:
                    self._validator.validate(val)
                value[num] = val
            except (TypeError, RuntimeError, yamlParserError, yamlScannerError, jsonschemaValidationError) as ex:
                elem = '' if not islist else ' element '+str(num+1)
                raise TypeError('Parser key "'+self.dest+'"'+elem+': '+str(ex)) from ex
        return value if islist else value[0]


    @staticmethod
    def split_ext_vars(
        ext_vars: Optional[Union[Dict[str, Any], Namespace]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Splits an ext_vars dict into the ext_codes and ext_vars required by jsonnet.

        Args:
            ext_vars: External variables. Values can be strings or any other basic type.
        """
        if ext_vars is None:
            ext_vars = {}
        elif isinstance(ext_vars, Namespace):
            ext_vars = namespace_to_dict(ext_vars)
        ext_codes = {k: json.dumps(v) for k, v in ext_vars.items() if not isinstance(v, str)}
        ext_vars = {k: v for k, v in ext_vars.items() if isinstance(v, str)}
        return ext_vars, ext_codes


    def parse(
        self,
        jsonnet: Union[str, Path],
        ext_vars: Union[Dict[str, Any], Namespace] = None,
        with_meta: bool = False,
    ) -> Namespace:
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
        _jsonnet = import_jsonnet('ActionJsonnet')
        ext_vars, ext_codes = self.split_ext_vars(ext_vars)
        fpath = None
        fname = 'snippet'
        snippet = jsonnet
        try:
            fpath = Path(jsonnet, mode=get_config_read_mode())
        except TypeError:
            pass
        else:
            fname = jsonnet(absolute=False) if isinstance(jsonnet, Path) else jsonnet
            snippet = fpath.get_content()
        try:
            values = yaml.safe_load(_jsonnet.evaluate_snippet(fname, snippet, ext_vars=ext_vars, ext_codes=ext_codes))
        except RuntimeError as ex:
            raise ParserError('Problems evaluating jsonnet "'+fname+'" :: '+str(ex)) from ex
        if self._validator is not None:
            self._validator.validate(values)
        if with_meta:
            if isinstance(values, dict) and fpath is not None:
                values['__path__'] = fpath
            values['__orig__'] = snippet
        return dict_to_namespace(values)
