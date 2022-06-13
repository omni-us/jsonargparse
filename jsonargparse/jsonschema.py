"""Action to support jsonschemas."""

import json
import os
from argparse import Action
from typing import Dict, Union

from .actions import _is_action_value_list
from .loaders_dumpers import get_loader_exceptions, load_value, load_value_context
from .namespace import strip_meta
from .optionals import (
    argcomplete_warn_redraw_prompt,
    get_jsonschema_exceptions,
    import_jsonschema,
)
from .util import parse_value_or_config


__all__ = ['ActionJsonSchema']


class ActionJsonSchema(Action):
    """Action to parse option as json validated by a jsonschema."""

    def __init__(
        self,
        schema: Union[str, Dict] = None,
        enable_path: bool = True,
        with_meta: bool = True,
        **kwargs
    ):
        """Initializer for ActionJsonSchema instance.

        Args:
            schema: Schema to validate values against.
            enable_path: Whether to try to load json from path (def.=True).
            with_meta: Whether to include metadata (def.=True).

        Raises:
            ValueError: If a parameter is invalid.
            jsonschema.exceptions.SchemaError: If the schema is invalid.
        """
        if schema is not None:
            if isinstance(schema, str):
                with load_value_context('yaml'):
                    try:
                        schema = load_value(schema)
                    except get_loader_exceptions() as ex:
                        raise ValueError(f'Problems parsing schema :: {ex}') from ex
            jsonvalidator = import_jsonschema('ActionJsonSchema')[1]
            jsonvalidator.check_schema(schema)
            self._validator = self._extend_jsonvalidator_with_default(jsonvalidator)(schema)
            self._enable_path = enable_path
            self._with_meta = with_meta
        elif '_validator' not in kwargs:
            raise ValueError('Expected schema keyword argument.')
        else:
            self._validator = kwargs.pop('_validator')
            self._enable_path = kwargs.pop('_enable_path')
            self._with_meta = kwargs.pop('_with_meta')
            super().__init__(**kwargs)


    def __call__(self, *args, **kwargs):
        """Parses an argument validating against the corresponding jsonschema.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs['_validator'] = self._validator
            kwargs['_enable_path'] = self._enable_path
            kwargs['_with_meta'] = self._with_meta
            if 'help' in kwargs and isinstance(kwargs['help'], str) and '%s' in kwargs['help']:
                kwargs['help'] = kwargs['help'] % json.dumps(self._validator.schema, sort_keys=True)
            return ActionJsonSchema(**kwargs)
        val = self._check_type(args[2])
        if not self._with_meta:
            val = strip_meta(val)
        setattr(args[1], self.dest, val)


    def _check_type(self, value, cfg=None):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        for num, val in enumerate(value):
            try:
                val, fpath = parse_value_or_config(val, enable_path=self._enable_path)
                path_meta = val.pop('__path__') if isinstance(val, dict) and '__path__' in val else None
                self._validator.validate(val)
                if path_meta is not None:
                    val['__path__'] = path_meta
                if isinstance(val, dict) and fpath is not None:
                    val['__path__'] = fpath
                value[num] = val
            except (TypeError, ValueError) + get_jsonschema_exceptions() + get_loader_exceptions() as ex:
                elem = '' if not islist else ' element '+str(num+1)
                raise TypeError(f'Parser key "{self.dest}"{elem}: {ex}') from ex
        return value if islist else value[0]


    @staticmethod
    def _extend_jsonvalidator_with_default(validator_class):
        """Extends a json schema validator so that it fills in default values."""
        validate_properties = validator_class.VALIDATORS['properties']

        def set_defaults(validator, properties, instance, schema):
            for prop, subschema in properties.items():
                if 'default' in subschema:
                    instance.setdefault(prop, subschema['default'])

            yield from validate_properties(validator, properties, instance, schema)

        jsonschema = import_jsonschema('ActionJsonSchema')[0]
        return jsonschema.validators.extend(validator_class, {'properties': set_defaults})


    def completer(self, prefix, **kwargs):
        """Used by argcomplete, validates value and shows expected type."""
        if chr(int(os.environ['COMP_TYPE'])) == '?':
            try:
                if prefix.strip() == '':
                    raise ValueError()
                self._validator.validate(load_value(prefix))
                msg = 'value already valid, '
            except (ValueError,) + get_jsonschema_exceptions() + get_loader_exceptions():
                msg = 'value not yet valid, '
            else:
                schema = json.dumps(self._validator.schema, indent=2, sort_keys=True).replace('\n', '\n  ')
                msg += f'required to be valid according to schema:\n  {schema}\n'
            return argcomplete_warn_redraw_prompt(prefix, msg)
