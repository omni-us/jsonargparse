"""Jsonschema functionalities."""

import json
import yaml
from argparse import Namespace, Action
from typing import Any, Union, Tuple, List, Set, Dict

from .util import namespace_to_dict, Path, strip_meta, _check_unknown_kwargs
from .optionals import _import_jsonschema, get_config_read_mode
from .actions import _is_action_value_list


class ActionJsonSchema(Action):
    """Action to parse option as json validated by a jsonschema."""

    def __init__(self, **kwargs):
        """Initializer for ActionJsonSchema instance.

        Args:
            schema (str or object): Schema to validate values against.
            annotation (type): Type object from which to generate schema.
            with_meta (bool): Whether to include metadata (def.=True).

        Raises:
            ValueError: If a parameter is invalid.
            jsonschema.exceptions.SchemaError: If the schema is invalid.
        """
        if 'schema' in kwargs or 'annotation' in kwargs:
            jsonvalidator = _import_jsonschema('ActionJsonSchema')[1]
            _check_unknown_kwargs(kwargs, {'schema', 'annotation', 'with_meta'})
            if 'annotation' in kwargs:
                if 'schema' in kwargs:
                    raise ValueError('Only one of schema or annotation is accepted.')
                self._annotation = kwargs['annotation']
                schema = ActionJsonSchema.typing_schema(kwargs['annotation'])
                if schema is None or schema == {'type': 'null'}:
                    raise ValueError('Unable to generate schema from annotation '+str(kwargs['annotation']))
            else:
                self._annotation = None
                schema = kwargs['schema']
            if isinstance(schema, str):
                try:
                    schema = yaml.safe_load(schema)
                except yaml.parser.ParserError as ex:
                    raise ValueError('Problems parsing schema :: '+str(ex))
            jsonvalidator.check_schema(schema)
            self._validator = self._extend_jsonvalidator_with_default(jsonvalidator)(schema)
            self._with_meta = kwargs.get('with_meta', True)
        elif '_validator' not in kwargs:
            raise ValueError('Expected schema or annotation keyword arguments.')
        else:
            self._annotation = kwargs.pop('_annotation')
            self._validator = kwargs.pop('_validator')
            self._with_meta = kwargs.pop('_with_meta')
            kwargs['type'] = str
            super().__init__(**kwargs)


    def __call__(self, *args, **kwargs):
        """Parses an argument validating against the corresponding jsonschema.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs['_annotation'] = self._annotation
            kwargs['_validator'] = self._validator
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
        jsonschema = _import_jsonschema('ActionJsonSchema')[0]
        if not islist:
            value = [value]
        elif not isinstance(value, list):
            raise TypeError('For ActionJsonSchema with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        for num, val in enumerate(value):
            try:
                fpath = None
                if isinstance(val, str):
                    val = yaml.safe_load(val)
                if isinstance(val, str):
                    try:
                        fpath = Path(val, mode=get_config_read_mode())
                    except:
                        pass
                    else:
                        val = yaml.safe_load(fpath.get_content())
                if isinstance(val, Namespace):
                    val = namespace_to_dict(val)
                if isinstance(val, (tuple, set)):
                    val = list(val)
                elif hasattr(self._annotation, '__origin__') and isinstance(val, dict):
                    val = self._adapt_dict(val, str)
                path_meta = val.pop('__path__') if isinstance(val, dict) and '__path__' in val else None
                self._validator.validate(val)
                if hasattr(self._annotation, '__origin__'):
                    if isinstance(val, dict):
                        val = self._adapt_dict(val, int)
                    elif self._annotation.__origin__ in {Tuple, tuple, Set, set} and isinstance(val, list):
                        val = tuple(val) if self._annotation.__origin__ in {Tuple, tuple} else set(val)
                if path_meta is not None:
                    val['__path__'] = path_meta
                if isinstance(val, dict) and fpath is not None:
                    val['__path__'] = fpath
                value[num] = val
            except (TypeError, yaml.parser.ParserError, jsonschema.exceptions.ValidationError) as ex:
                elem = '' if not islist else ' element '+str(num+1)
                raise TypeError('Parser key "'+self.dest+'"'+elem+': '+str(ex))
        return value if islist else value[0]


    def _adapt_dict(self, val, cast):
        def is_int_key(a):
            return hasattr(a, '__origin__') and a.__origin__ in {Dict, dict} and a.__args__[0] == int
        if is_int_key(self._annotation) or \
           (self._annotation.__origin__ == Union and any(is_int_key(a) for a in self._annotation.__args__)):
            try:
                val = {cast(k): v for k, v in val.items()}
            except ValueError:
                pass
        return val


    @staticmethod
    def _extend_jsonvalidator_with_default(validator_class):
        """Extends a json schema validator so that it fills in default values."""
        validate_properties = validator_class.VALIDATORS['properties']

        def set_defaults(validator, properties, instance, schema):
            for property, subschema in properties.items():
                if 'default' in subschema:
                    instance.setdefault(property, subschema['default'])

            for error in validate_properties(validator, properties, instance, schema):
                yield error

        jsonschema = _import_jsonschema('ActionJsonSchema')[0]
        return jsonschema.validators.extend(validator_class, {'properties': set_defaults})


    @staticmethod
    def typing_schema(annotation):
        """Generates a schema based on typing annotation."""
        typesmap = {
            str: 'string',
            int: 'integer',
            float: 'number',
            bool: 'boolean',
            type(None): 'null',
        }

        if annotation == Any:
            return {}

        elif annotation in typesmap:
            return {'type': typesmap[annotation]}

        elif not hasattr(annotation, '__origin__'):
            return

        elif annotation.__origin__ == Union:
            members = []
            for arg in annotation.__args__:
                schema = ActionJsonSchema.typing_schema(arg)
                if schema is not None:
                    members.append(schema)
            if len(members) == 1:
                return members[0]
            elif len(members) > 1:
                return {'anyOf': members}

        elif annotation.__origin__ in {Tuple, tuple}:
            items = [ActionJsonSchema.typing_schema(a) for a in annotation.__args__]
            if any(a is None for a in items):
                return
            return {'type': 'array', 'items': items}

        elif annotation.__origin__ in {List, list, Set, set}:
            items = ActionJsonSchema.typing_schema(annotation.__args__[0])
            if items is not None:
                return {'type': 'array', 'items': items}

        elif annotation.__origin__ in {Dict, dict} and annotation.__args__[0] in {str, int}:
            pattern = {str: '.*', int: '[0-9]+'}[annotation.__args__[0]]
            schema = ActionJsonSchema.typing_schema(annotation.__args__[1])
            if schema is not None:
                return {'type': 'object', 'patternProperties': {pattern: schema}}
