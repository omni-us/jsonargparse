"""Action to support jsonschema and type hint annotations."""

import os
import json
import yaml
import inspect
from enum import Enum
from argparse import Namespace, Action
from typing import Any, Union, Tuple, List, Iterable, Sequence, Set, Dict, Type

from .actions import _is_action_value_list
from .typing import is_optional, annotation_to_schema, type_to_str
from .util import (
    namespace_to_dict,
    Path,
    yamlParserError,
    yamlScannerError,
    ParserError,
    strip_meta,
    import_object,
    _load_config,
    _issubclass,
)
from .optionals import (
    ModuleNotFound,
    jsonschemaValidationError,
    import_jsonschema,
    files_completer,
    argcomplete_warn_redraw_prompt,
)


__all__ = ['ActionJsonSchema']


typesmap = {
    str: 'string',
    int: 'integer',
    float: 'number',
    bool: 'boolean',
    type(None): 'null',
}

supported_types = {
    bool,
    Any,
    Union,
    List, list, Iterable, Sequence,
    Tuple, tuple,
    Set, set,
    Dict, dict,
}


class ActionJsonSchema(Action):
    """Action to parse option as json validated by a jsonschema."""

    def __init__(
        self,
        schema: Union[str, Dict] = None,
        annotation: Type = None,
        enable_path: bool = True,
        with_meta: bool = True,
        **kwargs
    ):
        """Initializer for ActionJsonSchema instance.

        Args:
            schema: Schema to validate values against.
            annotation: Type object from which to generate schema.
            enable_path: Whether to try to load json from path (def.=True).
            with_meta: Whether to include metadata (def.=True).

        Raises:
            ValueError: If a parameter is invalid.
            jsonschema.exceptions.SchemaError: If the schema is invalid.
        """
        if schema is not None or annotation is not None:
            if annotation is not None:
                if schema is not None:
                    raise ValueError('Only one of schema or annotation is accepted.')
                self._annotation = annotation
                schema, subschemas = ActionJsonSchema._typing_schema(self._annotation)
                if schema is None or schema == {'type': 'null'}:
                    raise ValueError('Unable to generate schema from annotation '+str(self._annotation))
                self._subschemas = subschemas
            else:
                self._annotation = self._subschemas = None  # type: ignore
            if isinstance(schema, str):
                try:
                    schema = yaml.safe_load(schema)
                except (yamlParserError, yamlScannerError) as ex:
                    raise ValueError('Problems parsing schema :: '+str(ex)) from ex
            jsonvalidator = import_jsonschema('ActionJsonSchema')[1]
            jsonvalidator.check_schema(schema)
            self._validator = self._extend_jsonvalidator_with_default(jsonvalidator)(schema)
            self._enable_path = enable_path
            self._with_meta = with_meta
        elif '_validator' not in kwargs:
            raise ValueError('Expected schema or annotation keyword arguments.')
        else:
            self._annotation = kwargs.pop('_annotation')
            self._subschemas = kwargs.pop('_subschemas')
            self._validator = kwargs.pop('_validator')
            self._enable_path = kwargs.pop('_enable_path')
            self._with_meta = kwargs.pop('_with_meta')
            metavar = self._annotation_metavar()
            if metavar is not None:
                kwargs['metavar'] = metavar
            super().__init__(**kwargs)


    def __call__(self, *args, **kwargs):
        """Parses an argument validating against the corresponding jsonschema.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs['_annotation'] = self._annotation
            kwargs['_subschemas'] = self._subschemas
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
                val, fpath = _load_config(val, enable_path=self._enable_path, flat_namespace=False)
                if isinstance(val, Namespace):
                    val = namespace_to_dict(val)
                val = self._adapt_types(val, self._annotation, self._subschemas, reverse=True)
                path_meta = val.pop('__path__') if isinstance(val, dict) and '__path__' in val else None
                self._validator.validate(val)
                val = self._adapt_types(val, self._annotation, self._subschemas)
                if path_meta is not None:
                    val['__path__'] = path_meta
                if isinstance(val, dict) and fpath is not None:
                    val['__path__'] = fpath
                value[num] = val
            except (TypeError, yamlParserError, yamlScannerError, jsonschemaValidationError) as ex:
                elem = '' if not islist else ' element '+str(num+1)
                raise TypeError('Parser key "'+self.dest+'"'+elem+': '+str(ex)) from ex
        return value if islist else value[0]


    @staticmethod
    def _extend_jsonvalidator_with_default(validator_class):
        """Extends a json schema validator so that it fills in default values."""
        validate_properties = validator_class.VALIDATORS['properties']

        def set_defaults(validator, properties, instance, schema):
            for prop, subschema in properties.items():
                if 'default' in subschema:
                    instance.setdefault(prop, subschema['default'])

            for error in validate_properties(validator, properties, instance, schema):
                yield error

        jsonschema = import_jsonschema('ActionJsonSchema')[0]
        return jsonschema.validators.extend(validator_class, {'properties': set_defaults})


    def _instantiate_classes(self, val):
        if self._annotation is not None:
            val = self._adapt_types(val, self._annotation, self._subschemas, instantiate_classes=True)
        return val


    @staticmethod
    def _adapt_types(val, annotation, subschemas, reverse=False, instantiate_classes=False):

        def validate_adapt(v, subschema):
            if subschema is not None:
                subannotation, subvalidator, subsubschemas = subschema
                if reverse:
                    v = ActionJsonSchema._adapt_types(v, subannotation, subsubschemas, reverse, instantiate_classes)
                else:
                    try:
                        if subvalidator is not None and not instantiate_classes:
                            subvalidator.validate(v)
                        v = ActionJsonSchema._adapt_types(v, subannotation, subsubschemas, reverse, instantiate_classes)
                    except jsonschemaValidationError:
                        pass
            return v

        if subschemas is None:
            subschemas = []

        if _issubclass(annotation, Enum):
            if reverse and isinstance(val, annotation):
                val = val.name
            elif not reverse and val in annotation.__members__:
                val = annotation[val]

        elif _issubclass(annotation, Path):
            if reverse and isinstance(val, annotation):
                val = str(val)
            elif not reverse:
                val = annotation(val)

        elif not hasattr(annotation, '__origin__'):
            if not reverse and \
               not _issubclass(annotation, (str, int, float)) and \
               isinstance(val, dict) and \
               'class_path' in val:
                try:
                    val_class = import_object(val['class_path'])
                    assert _issubclass(val_class, annotation), 'Not a subclass of '+annotation.__name__
                    if 'init_args' in val:
                        from jsonargparse import ArgumentParser
                        parser = ArgumentParser(error_handler=None, parse_as_dict=True)
                        parser.add_class_arguments(val_class)
                        parser.check_config(val['init_args'])
                        if instantiate_classes:
                            init_args = parser.instantiate_subclasses(val['init_args'])
                            val = val_class(**init_args)  # pylint: disable=not-a-mapping
                    elif instantiate_classes:
                        val = val_class()
                except (ImportError, ModuleNotFound, AttributeError, AssertionError, ParserError) as ex:
                    raise ParserError('Problem with given class_path "'+val['class_path']+'" :: '+str(ex)) from ex
            return val

        elif annotation.__origin__ == Union:
            for subschema in subschemas:
                val = validate_adapt(val, subschema)

        elif annotation.__origin__ in {Tuple, tuple, Set, set} and isinstance(val, (list, tuple, set)):
            if reverse:
                val = list(val)
            for n, v in enumerate(val):
                if len(subschemas) == 0:
                    break
                subschema = subschemas[n if n < len(subschemas) else -1]
                if subschema is not None:
                    val[n] = validate_adapt(v, subschema)
            if not reverse:
                val = tuple(val) if annotation.__origin__ in {Tuple, tuple} else set(val)

        elif annotation.__origin__ in {List, list, Set, set, Iterable, Sequence} and isinstance(val, list):
            for n, v in enumerate(val):
                for subschema in subschemas:
                    val[n] = validate_adapt(v, subschema)

        elif annotation.__origin__ in {Dict, dict} and isinstance(val, dict):
            if annotation.__args__[0] == int:
                cast = str if reverse else int
                val = {cast(k): v for k, v in val.items()}
            if annotation.__args__[1] not in typesmap:
                for k, v in val.items():
                    for subschema in subschemas:
                        val[k] = validate_adapt(v, subschema)

        return val


    @staticmethod
    def _typing_schema(annotation):
        """Generates a schema based on a type annotation."""

        if annotation == Any:
            return {}, None

        elif annotation in typesmap:
            return {'type': typesmap[annotation]}, None

        elif _issubclass(annotation, Enum):
            return {'type': 'string', 'enum': list(annotation.__members__.keys())}, [(annotation, None, None)]

        elif _issubclass(annotation, Path):
            return {'type': 'string'}, [(annotation, None, None)]

        elif _issubclass(annotation, (str, int, float)):
            return annotation_to_schema(annotation), None

        elif annotation == dict:
            return {'type': 'object'}, None

        elif not hasattr(annotation, '__origin__'):
            if annotation != inspect._empty:
                schema = {
                    'type': 'object',
                    'properties': {
                        'class_path': {'type': 'string'},
                        'init_args': {'type': 'object'},
                    },
                    'required': ['class_path'],
                    'additionalProperties': False,
                }
                jsonvalidator = import_jsonschema('ActionJsonSchema')[1]
                return schema, [(annotation, jsonvalidator(schema), None)]
            return None, None

        elif annotation.__origin__ == Union:
            members = []
            union_subschemas = []
            for arg in annotation.__args__:
                schema, subschemas = ActionJsonSchema._typing_schema(arg)
                if schema is not None:
                    members.append(schema)
                    if arg not in typesmap:
                        jsonvalidator = import_jsonschema('ActionJsonSchema')[1]
                        union_subschemas.append((arg, jsonvalidator(schema), subschemas))
            if len(members) == 1:
                return members[0], union_subschemas
            elif len(members) > 1:
                return {'anyOf': members}, union_subschemas

        elif annotation.__origin__ in {Tuple, tuple}:
            has_ellipsis = False
            items = []
            tuple_subschemas = []
            for arg in annotation.__args__:
                if arg == Ellipsis:
                    has_ellipsis = True
                    break
                item, subschemas = ActionJsonSchema._typing_schema(arg)
                items.append(item)
                if arg not in typesmap:
                    jsonvalidator = import_jsonschema('ActionJsonSchema')[1]
                    tuple_subschemas.append((arg, jsonvalidator(item), subschemas))
            schema = {'type': 'array', 'items': items, 'minItems': len(items)}
            if has_ellipsis:
                schema['additionalItems'] = items[-1]
            else:
                schema['maxItems'] = len(items)
            return schema, tuple_subschemas

        elif annotation.__origin__ in {List, list, Iterable, Sequence, Set, set}:
            items, subschemas = ActionJsonSchema._typing_schema(annotation.__args__[0])
            if items is not None:
                return {'type': 'array', 'items': items}, subschemas

        elif annotation.__origin__ in {Dict, dict} and annotation.__args__[0] in {str, int}:
            pattern = {str: '.*', int: '[0-9]+'}[annotation.__args__[0]]
            schema, subschemas = ActionJsonSchema._typing_schema(annotation.__args__[1])
            if schema is not None:
                return {'type': 'object', 'patternProperties': {pattern: schema}}, subschemas

        return None, None


    def _annotation_metavar(self):
        """Generates a metavar for some types."""
        metavar = None
        if self._annotation == bool:
            metavar = '{true,false}'
        elif is_optional(self._annotation, bool):
            metavar = '{true,false,null}'
        elif is_optional(self._annotation, Enum):
            enum = self._annotation.__args__[0]
            metavar = '{'+','.join(list(enum.__members__.keys())+['null'])+'}'
        return metavar


    def completer(self, prefix, **kwargs):
        """Used by argcomplete, validates value and shows expected type."""
        if self._annotation == bool:
            return ['true', 'false']
        elif is_optional(self._annotation, bool):
            return ['true', 'false', 'null']
        elif is_optional(self._annotation, Enum):
            enum = self._annotation.__args__[0]
            return list(enum.__members__.keys())+['null']
        elif is_optional(self._annotation, Path):
            return ['null'] + sorted(files_completer(prefix, **kwargs))
        elif chr(int(os.environ['COMP_TYPE'])) == '?':
            try:
                if prefix.strip() == '':
                    raise ValueError()
                self._validator.validate(yaml.safe_load(prefix))
                msg = 'value already valid, '
            except (ValueError, yamlParserError, yamlScannerError, jsonschemaValidationError):
                msg = 'value not yet valid, '
            if self._annotation is not None:
                msg += 'expected type '+type_to_str(self._annotation)
            else:
                schema = json.dumps(self._validator.schema, indent=2, sort_keys=True).replace('\n', '\n  ')
                msg += 'required to be valid according to schema:\n  '+schema+'\n'
            return argcomplete_warn_redraw_prompt(prefix, msg)
