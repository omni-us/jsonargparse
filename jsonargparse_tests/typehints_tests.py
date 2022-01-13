#!/usr/bin/env python3

import json
import os
import pathlib
import sys
import unittest
import uuid
import yaml
from calendar import Calendar
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union
from jsonargparse import ActionConfigFile, ArgumentParser, CLI, lazy_instance, Namespace, ParserError, Path
from jsonargparse.typehints import ActionTypeHint, is_optional, Literal
from jsonargparse.typing import (
    Email,
    NotEmptyStr,
    OpenUnitInterval,
    Path_drw,
    Path_fc,
    Path_fr,
    path_type,
    PositiveInt,
    register_type,
)
from jsonargparse_tests.base import mock_module, TempDirTestCase


class TypeHintsTests(unittest.TestCase):

    def test_add_argument_type_hint(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--op1', type=Optional[Union[PositiveInt, OpenUnitInterval]])
        self.assertEqual(0.1, parser.parse_args(['--op1', '0.1']).op1)
        self.assertEqual(0.9, parser.parse_args(['--op1', '0.9']).op1)
        self.assertEqual(1, parser.parse_args(['--op1', '1']).op1)
        self.assertEqual(12, parser.parse_args(['--op1', '12']).op1)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op1', '0.0']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op1', '4.5']))
        parser.add_argument('--op2', type=Optional[Email])
        self.assertEqual('a@b.c', parser.parse_args(['--op2', 'a@b.c']).op2)
        self.assertIsNone(parser.parse_args(['--op2=null']).op2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op2', 'abc']))


    def test_type_hint_action_failure(self):
        parser = ArgumentParser(error_handler=None)
        self.assertRaises(ValueError, lambda: parser.add_argument('--op1', type=Optional[bool], action=True))


    def test_bool(self):
        parser = ArgumentParser(prog='app', default_env=True, error_handler=None)
        parser.add_argument('--val', type=bool)
        self.assertEqual(None,  parser.get_defaults().val)
        self.assertEqual(True,  parser.parse_args(['--val', 'true']).val)
        self.assertEqual(True,  parser.parse_args(['--val', 'TRUE']).val)
        self.assertEqual(False, parser.parse_args(['--val', 'false']).val)
        self.assertEqual(False, parser.parse_args(['--val', 'FALSE']).val)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--val', '1']))

        os.environ['APP_VAL'] = 'true'
        self.assertEqual(True,  parser.parse_args([]).val)
        os.environ['APP_VAL'] = 'True'
        self.assertEqual(True,  parser.parse_args([]).val)
        os.environ['APP_VAL'] = 'false'
        self.assertEqual(False, parser.parse_args([]).val)
        os.environ['APP_VAL'] = 'False'
        self.assertEqual(False, parser.parse_args([]).val)
        os.environ['APP_VAL'] = '2'
        self.assertRaises(ParserError, lambda: parser.parse_args(['--val', 'a']))
        del os.environ['APP_VAL']


    def test_no_str_strip(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--op', type=Optional[str])
        parser.add_argument('--cfg', action=ActionConfigFile)
        self.assertEqual('  ', parser.parse_args(['--op', '  ']).op)
        self.assertEqual('', parser.parse_args(['--op', '']).op)
        self.assertEqual(' abc ', parser.parse_args(['--op= abc ']).op)
        self.assertEqual(' ', parser.parse_args(['--cfg={"op":" "}']).op)
        self.assertIsNone(parser.parse_args(['--op=null']).op)


    def test_list(self):
        for list_type in [Iterable, List, Sequence]:
            with self.subTest(str(list_type)):
                parser = ArgumentParser()
                parser.add_argument('--list', type=list_type[int])
                cfg = parser.parse_args(['--list=[1, 2]'])
                self.assertEqual([1, 2], cfg.list)


    def test_list_enum(self):
        class MyEnum(Enum):
            ab = 0
            xy = 1

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--list', type=List[MyEnum])
        self.assertEqual([MyEnum.xy, MyEnum.ab], parser.parse_args(['--list=["xy", "ab"]']).list)


    def test_list_union(self):
        class MyEnum(Enum):
            ab = 1

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--list1', type=List[Union[float, str, type(None)]])
        parser.add_argument('--list2', type=List[Union[int, MyEnum]])
        self.assertEqual([1.2, 'ab'], parser.parse_args(['--list1=[1.2, "ab"]']).list1)
        self.assertEqual([3, MyEnum.ab], parser.parse_args(['--list2=[3, "ab"]']).list2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--list1={"a":1, "b":"2"}']))


    def test_dict(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--dict', type=dict)
        self.assertEqual({}, parser.parse_args(['--dict={}'])['dict'])
        self.assertEqual({'a': 1, 'b': '2'}, parser.parse_args(['--dict={"a":1, "b":"2"}'])['dict'])
        self.assertRaises(ParserError, lambda: parser.parse_args(['--dict=1']))


    def test_dict_union(self):
        class MyEnum(Enum):
            ab = 1

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--dict1', type=Dict[int, Optional[Union[float, MyEnum]]])
        parser.add_argument('--dict2', type=Dict[str, Union[bool, Path_fc]])
        cfg = parser.parse_args(['--dict1={"2":4.5, "6":"ab"}', '--dict2={"a":true, "b":"f"}'])
        self.assertEqual({2: 4.5, 6: MyEnum.ab}, cfg['dict1'])
        self.assertEqual({'a': True, 'b': 'f'}, cfg['dict2'])
        self.assertIsInstance(cfg['dict2']['b'], Path)
        self.assertEqual({5: None}, parser.parse_args(['--dict1={"5":null}'])['dict1'])
        self.assertRaises(ParserError, lambda: parser.parse_args(['--dict1=["a", "b"]']))
        cfg = yaml.safe_load(parser.dump(cfg))
        self.assertEqual({'dict1': {'2': 4.5, '6': 'ab'}, 'dict2': {'a': True, 'b': 'f'}}, cfg)


    def test_tuple(self):
        class MyEnum(Enum):
            ab = 1

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--tuple', type=Tuple[Union[int, MyEnum], Path_fc, NotEmptyStr])
        cfg = parser.parse_args(['--tuple=[2, "a", "b"]'])
        self.assertEqual((2, 'a', 'b'), cfg.tuple)
        self.assertIsInstance(cfg.tuple[1], Path)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=[]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=[2, "a", "b", 5]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=[2, "a"]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=["2", "a", "b"]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple={"a":1, "b":"2"}']))


    def test_nested_tuples(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--tuple', type=Tuple[Tuple[str, str], Tuple[Tuple[int, float], Tuple[int, float]]])
        cfg = parser.parse_args(['--tuple=[["foo", "bar"], [[1, 2.02], [3, 3.09]]]'])
        self.assertEqual((('foo', 'bar'), ((1, 2.02), (3, 3.09))), cfg.tuple)


    def test_list_tuple(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--list', type=List[Tuple[int, float]])
        cfg = parser.parse_args(['--list=[[1, 2.02], [3, 3.09]]'])
        self.assertEqual([(1, 2.02), (3, 3.09)], cfg.list)


    def test_tuple_ellipsis(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--tuple', type=Tuple[float, ...])
        self.assertEqual((1.2,), parser.parse_args(['--tuple=[1.2]']).tuple)
        self.assertEqual((1.2, 3.4), parser.parse_args(['--tuple=[1.2, 3.4]']).tuple)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=[]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=[2, "a"]']))

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--tuple', type=Tuple[Tuple[str, str], Tuple[Tuple[int, float], ...]])
        cfg = parser.parse_args(['--tuple=[["foo", "bar"], [[1, 2.02], [3, 3.09]]]'])
        self.assertEqual((('foo', 'bar'), ((1, 2.02), (3, 3.09))), cfg.tuple)


    def test_complex_number(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--complex', type=complex)
        cfg = parser.parse_args(['--complex=(2+3j)'])
        self.assertEqual(cfg.complex, 2+3j)
        self.assertEqual(parser.dump(cfg), 'complex: (2+3j)\n')


    def test_type_Any(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--any', type=Any)
        self.assertEqual('abc', parser.parse_args(['--any=abc'])['any'])
        self.assertEqual(123, parser.parse_args(['--any=123'])['any'])
        self.assertEqual(5.6, parser.parse_args(['--any=5.6'])['any'])
        self.assertEqual([7, 8], parser.parse_args(['--any=[7, 8]'])['any'])
        self.assertEqual({"a":0, "b":1}, parser.parse_args(['--any={"a":0, "b":1}'])['any'])
        self.assertTrue(parser.parse_args(['--any=True'])['any'])
        self.assertFalse(parser.parse_args(['--any=False'])['any'])
        self.assertIsNone(parser.parse_args(['--any=null'])['any'])
        self.assertEqual(' ', parser.parse_args(['--any= '])['any'])
        self.assertEqual(' xyz ', parser.parse_args(['--any= xyz '])['any'])
        self.assertEqual('[[[', parser.parse_args(['--any=[[['])['any'])


    @unittest.skipIf(sys.version_info[:2] < (3, 8), 'Literal introduced in python 3.8')
    def test_Literal(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--str', type=Literal['a', 'b'])
        parser.add_argument('--true', type=Literal[True])
        parser.add_argument('--false', type=Literal[False])
        self.assertEqual('a', parser.parse_args(['--str=a']).str)
        self.assertEqual('b', parser.parse_args(['--str=b']).str)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--str=x']))
        self.assertIs(True, parser.parse_args(['--true=true']).true)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--true=false']))
        self.assertIs(False, parser.parse_args(['--false=false']).false)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--false=true']))


    def _test_typehint_non_parameterized_types(self, type):
        parser = ArgumentParser(error_handler=None)
        ActionTypeHint.is_supported_typehint(type, full=True)
        parser.add_argument('--type', type=type)
        cfg = parser.parse_args(['--type=uuid.UUID'])
        self.assertEqual(cfg.type, uuid.UUID)
        self.assertEqual(parser.dump(cfg), 'type: uuid.UUID\n')


    def _test_typehint_parameterized_types(self, type):
        parser = ArgumentParser(error_handler=None)
        ActionTypeHint.is_supported_typehint(type, full=True)
        parser.add_argument('--cal', type=type[Calendar])
        cfg = parser.parse_args(['--cal=calendar.Calendar'])
        self.assertEqual(cfg.cal, Calendar)
        self.assertEqual(parser.dump(cfg), 'cal: calendar.Calendar\n')
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cal=uuid.UUID']))


    def test_typehint_Type(self):
        self._test_typehint_non_parameterized_types(type=Type)
        self._test_typehint_parameterized_types(type=Type)


    def test_typehint_non_parameterized_type(self):
        self._test_typehint_non_parameterized_types(type=type)


    @unittest.skipIf(sys.version_info[:2] < (3, 9), '[] support for builtins introduced in python 3.9')
    def test_typehint_parametrized_type(self):
        self._test_typehint_parameterized_types(type=type)


    def test_uuid(self):
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--uuid', type=uuid.UUID)
        parser.add_argument('--uuids', type=List[uuid.UUID])
        cfg = parser.parse_args(['--uuid='+str(id1), '--uuids=["'+str(id1)+'", "'+str(id2)+'"]'])
        self.assertEqual(cfg.uuid, id1)
        self.assertEqual(cfg.uuids, [id1, id2])
        self.assertEqual('uuid: '+str(id1)+'\nuuids:\n- '+str(id1)+'\n- '+str(id2)+'\n', parser.dump(cfg))


    def test_Callable(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--callable', type=Callable)
        parser.add_argument('--list', type=List[Callable])

        cfg = parser.parse_args(['--callable=jsonargparse.CLI'])
        self.assertEqual(CLI, cfg.callable)
        self.assertEqual(parser.dump(cfg), 'callable: jsonargparse.cli.CLI\n')
        self.assertEqual([CLI], parser.parse_args(['--list=[jsonargparse.CLI]']).list)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--callable=jsonargparse.not_exist']))


    def test_class_type(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--op', type=Optional[List[Calendar]])

        class_path = '"class_path": "calendar.Calendar"'
        expected = [{'class_path': 'calendar.Calendar', 'init_args': {'firstweekday': 0}}]
        cfg = parser.parse_args(['--op=[{'+class_path+'}]'])
        self.assertEqual(cfg.as_dict()['op'], expected)
        cfg = parser.parse_args(['--op=["calendar.Calendar"]'])
        self.assertEqual(cfg.as_dict()['op'], expected)
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['op'][0], Calendar)

        with self.assertRaises(ParserError):
            parser.parse_args(['--op=[{"class_path": "jsonargparse.ArgumentParser"}]'])
        with self.assertRaises(ParserError):
            parser.parse_args(['--op=[{"class_path": "jsonargparse.NotExist"}]'])
        with self.assertRaises(ParserError):
            parser.parse_args(['--op=[{"class_path": "jsonargparse0.IncorrectModule"}]'])
        with self.assertRaises(ParserError):
            parser.parse_args(['--op=[1]'])

        init_args = '"init_args": {"bad_arg": True}'
        with self.assertRaises(ParserError):
            parser.parse_args(['--op=[{'+class_path+', '+init_args+'}]'])

        init_args = '"init_args": {"firstweekday": 3}'
        cfg = parser.parse_args(['--op=[{'+class_path+', '+init_args+'}]'])
        self.assertEqual(cfg['op'][0]['init_args'].as_dict(), {'firstweekday': 3})
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['op'][0], Calendar)
        self.assertEqual(3, cfg['op'][0].firstweekday)

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--n.op', type=Optional[Calendar])
        cfg = parser.parse_args(['--n.op={'+class_path+', '+init_args+'}'])
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['n']['op'], Calendar)
        self.assertEqual(3, cfg['n']['op'].firstweekday)

        parser = ArgumentParser()
        parser.add_argument('--op', type=Calendar)
        cfg = parser.parse_args(['--op={'+class_path+', '+init_args+'}'])
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['op'], Calendar)
        self.assertEqual(3, cfg['op'].firstweekday)

        cfg = parser.instantiate_classes(parser.parse_args([]))
        self.assertIsNone(cfg['op'])


    def test_class_type_without_defaults(self):
        class MyCal(Calendar):
            def __init__(self, p1: int = 1, p2: str = '2'):
                pass

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--op', type=MyCal)

        with mock_module(MyCal) as module:
            cfg = parser.parse_args([f'--op.class_path={module}.MyCal', '--op.init_args.p1=3'], defaults=False)
            self.assertEqual(cfg.op, Namespace(class_path=f'{module}.MyCal', init_args=Namespace(p1=3)))


    def test_class_type_required_params(self):
        class MyCal(Calendar):
            def __init__(self, p1: int, p2: str):
                pass

        with mock_module(MyCal) as module:
            parser = ArgumentParser(error_handler=None)
            parser.add_argument('--op', type=MyCal, default=lazy_instance(MyCal))

            cfg = parser.get_defaults()
            self.assertEqual(cfg.op.class_path, f'{module}.MyCal')
            self.assertEqual(cfg.op.init_args, Namespace(p1=None, p2=None))


    def test_invalid_init_args_in_yaml(self):
        config = """cal:
            class_path: calendar.Calendar
            init_args:
        """
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--config', action=ActionConfigFile)
        parser.add_argument('--cal', type=Calendar)
        self.assertRaises(ParserError, lambda: parser.parse_args([f'--config={config}']))


    def test_typehint_serialize_list(self):
        parser = ArgumentParser()
        action = parser.add_argument('--list', type=Union[PositiveInt, List[PositiveInt]])
        self.assertEqual([1, 2], action.serialize([PositiveInt(1), PositiveInt(2)]))
        self.assertRaises(ValueError, lambda: action.serialize([1, -2]))


    def test_typehint_serialize_enum(self):

        class MyEnum(Enum):
            a = 1
            b = 2

        parser = ArgumentParser()
        action = parser.add_argument('--enum', type=Optional[MyEnum])
        self.assertEqual('b', action.serialize(MyEnum.b))
        self.assertRaises(ValueError, lambda: action.serialize('x'))


    def test_unsupported_type(self):
        self.assertRaises(ValueError, lambda: ActionTypeHint(typehint=lambda: None))
        self.assertRaises(ValueError, lambda: ActionTypeHint(typehint=Union[int, lambda: None]))


    def test_nargs_questionmark(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('p1')
        parser.add_argument('p2', nargs='?', type=OpenUnitInterval)
        self.assertIsNone(parser.parse_args(['a']).p2)
        self.assertEqual(0.5, parser.parse_args(['a', '0.5']).p2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['a', 'b']))


    def test_register_type(self):

        def serializer(v):
            return v.isoformat()

        def deserializer(v):
            return datetime.strptime(v, '%Y-%m-%dT%H:%M:%S')

        register_type(datetime, serializer, deserializer)
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--datetime', type=datetime)
        cfg = parser.parse_args(['--datetime=2008-09-03T20:56:35'])
        self.assertEqual(cfg.datetime, datetime(2008, 9, 3, 20, 56, 35))
        self.assertEqual(parser.dump(cfg), "datetime: '2008-09-03T20:56:35'\n")
        self.assertRaises(ValueError, lambda: register_type(datetime))
        register_type(uuid.UUID)


    def test_lazy_instance_invalid_kwargs(self):
        class MyClass:
            def __init__(self, param: int = 1):
                pass

        self.assertRaises(ValueError, lambda: lazy_instance(MyClass, param='bad'))


class TypeHintsTmpdirTests(TempDirTestCase):

    def test_path(self):
        os.mkdir(os.path.join(self.tmpdir, 'example'))
        rel_yaml_file = os.path.join('..', 'example', 'example.yaml')
        abs_yaml_file = os.path.realpath(os.path.join(self.tmpdir, 'example', rel_yaml_file))
        with open(abs_yaml_file, 'w') as output_file:
            output_file.write('file: '+rel_yaml_file+'\ndir: '+self.tmpdir+'\n')

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--cfg', action=ActionConfigFile)
        parser.add_argument('--file', type=Path_fr)
        parser.add_argument('--dir', type=Path_drw)
        parser.add_argument('--files', nargs='+', type=Path_fr)

        cfg = parser.parse_args(['--cfg', abs_yaml_file])
        self.assertEqual(self.tmpdir, os.path.realpath(cfg.dir()))
        self.assertEqual(rel_yaml_file, str(cfg.file))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file()))

        cfg = parser.parse_args(['--cfg', 'file: '+abs_yaml_file+'\ndir: '+self.tmpdir+'\n'])
        self.assertEqual(self.tmpdir, os.path.realpath(cfg.dir()))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file()))

        cfg = parser.parse_args(['--file', abs_yaml_file, '--dir', self.tmpdir])
        self.assertEqual(self.tmpdir, os.path.realpath(cfg.dir()))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file()))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--dir', abs_yaml_file]))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--file', self.tmpdir]))

        cfg = parser.parse_args(['--files', abs_yaml_file, abs_yaml_file])
        self.assertTrue(isinstance(cfg.files, list))
        self.assertEqual(2, len(cfg.files))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.files[-1]()))


    def test_list_path(self):
        parser = ArgumentParser()
        parser.add_argument('--paths', type=List[Path_fc])
        cfg = parser.parse_args(['--paths=["file1", "file2"]'])
        self.assertEqual(['file1', 'file2'], cfg.paths)
        self.assertIsInstance(cfg.paths[0], Path)
        self.assertIsInstance(cfg.paths[1], Path)


    def test_optional_path(self):
        pathlib.Path('file_fr').touch()
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--path', type=Optional[Path_fr])
        self.assertIsNone(parser.parse_args(['--path=null']).path)
        cfg = parser.parse_args(['--path=file_fr'])
        self.assertEqual('file_fr', cfg.path)
        self.assertIsInstance(cfg.path, Path)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--path=not_exist']))


    def test_enable_path(self):
        data = {'a': 1, 'b': 2, 'c': [3, 4]}
        cal = {'class_path': 'calendar.Calendar'}
        with open('data.yaml', 'w') as f:
            json.dump(data, f)
        with open('cal.yaml', 'w') as f:
            json.dump(cal, f)

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--data', type=Dict[str, Any], enable_path=True)
        parser.add_argument('--cal', type=Calendar, enable_path=True)
        cfg = parser.parse_args(['--data=data.yaml'])
        self.assertEqual('data.yaml', str(cfg['data'].pop('__path__')))
        self.assertEqual(data, cfg['data'])
        cfg = parser.instantiate_classes(parser.parse_args(['--cal=cal.yaml']))
        self.assertIsInstance(cfg['cal'], Calendar)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--data=does-not-exist.yaml']))


    def test_default_path_unregistered_type(self):
        parser = ArgumentParser()
        parser.add_argument('--path',
                            type=path_type('drw', skip_check=True),
                            default=Path('test', mode='drw', skip_check=True))
        cfg = parser.parse_args([])
        self.assertEqual('path: test\n', parser.dump(cfg))


    def test_class_type_with_default_config_files(self):
        config = {
            'class_path': 'calendar.Calendar',
            'init_args': {'firstweekday': 3},
        }
        config_path = os.path.join(self.tmpdir, 'config.yaml')
        with open(config_path, 'w') as f:
            json.dump({'data': {'cal': config}}, f)

        class MyClass:
            def __init__(self, cal: Optional[Calendar] = None, val: int = 2):
                self.cal = cal

        parser = ArgumentParser(error_handler=None, default_config_files=[config_path])
        parser.add_argument('--op', default='from default')
        parser.add_class_arguments(MyClass, 'data')

        cfg = parser.get_defaults()
        self.assertEqual(config_path, str(cfg['__default_config__']))
        self.assertEqual(cfg.data.cal.as_dict(), config)
        dump = parser.dump(cfg)
        self.assertIn('class_path: calendar.Calendar\n', dump)
        self.assertIn('firstweekday: 3\n', dump)

        cfg = parser.parse_args([])
        self.assertEqual(cfg.data.cal.as_dict(), config)
        cfg = parser.parse_args(['--data.cal.class_path=calendar.Calendar'], defaults=False)
        self.assertEqual(cfg.data.cal, Namespace(class_path='calendar.Calendar'))


    def test_class_path_override_with_default_config_files(self):

        class MyCalendar(Calendar):
            def __init__(self, *args, param: str = '0', **kwargs):
                super().__init__(*args, **kwargs)

        with mock_module(MyCalendar) as module:
            config = {
                'class_path': f'{module}.MyCalendar',
                'init_args': {'firstweekday': 2, 'param': '1'},
            }
            config_path = os.path.join(self.tmpdir, 'config.yaml')
            with open(config_path, 'w') as f:
                json.dump({'cal': config}, f)

            parser = ArgumentParser(error_handler=None, default_config_files=[config_path])
            parser.add_argument('--cal', type=Optional[Calendar])

            cfg = parser.instantiate_classes(parser.get_defaults())
            self.assertIsInstance(cfg['cal'], MyCalendar)

            cfg = parser.parse_args(['--cal={"class_path": "calendar.Calendar", "init_args": {"firstweekday": 3}}'])
            self.assertEqual(type(parser.instantiate_classes(cfg)['cal']), Calendar)


    def test_linking_deep_targets(self):
        class D:
            pass

        class A:
            def __init__(self, d: D) -> None:
                self.d = d

        class BSuper:
            pass

        class BSub(BSuper):
            def __init__(self, a: A) -> None:
                self.a = a

        class C:
            def fn(self) -> D:
                return D()

        with mock_module(D, A, BSuper, BSub, C) as module:
            config = {
                "b": {
                    "class_path": f"{module}.BSub",
                    "init_args": {
                        "a": {
                            "class_path": f"{module}.A",
                        },
                    },
                },
                "c": {},
            }
            config_path = os.path.join(self.tmpdir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f)

            parser = ArgumentParser()
            parser.add_argument("--config", action=ActionConfigFile)
            parser.add_subclass_arguments(BSuper, nested_key="b", required=True)
            parser.add_class_arguments(C, nested_key="c")
            parser.link_arguments("c", "b.init_args.a.init_args.d", compute_fn=C.fn, apply_on="instantiate")

            config = parser.parse_args(["--config", config_path])
            config_init = parser.instantiate_classes(config)
            self.assertIsInstance(config_init["b"].a.d, D)


    def test_mapping_class_typehint(self):
        class A:
            pass

        class B:
            def __init__(
                self,
                class_map: Mapping[str, A],
                int_list: List[int],
            ):
                self.class_map = class_map
                self.int_list = int_list

        with mock_module(A, B) as module:
            parser = ArgumentParser(error_handler=None)
            parser.add_class_arguments(B, 'b')

            config = {
                'b': {
                    'class_map': {
                        'one': {'class_path': f'{module}.A'},
                    },
                    'int_list': [1],
                },
            }

            cfg = parser.parse_object(config)
            self.assertEqual(cfg.b.class_map, {'one': Namespace(class_path=f'{module}.A')})
            self.assertEqual(cfg.b.int_list, [1])

            cfg_init = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg_init.b, B)
            self.assertIsInstance(cfg_init.b.class_map, dict)
            self.assertIsInstance(cfg_init.b.class_map['one'], A)

            config['b']['int_list'] = config['b']['class_map']
            self.assertRaises(ParserError, lambda: parser.parse_object(config))


    def test_linking_deep_targets_mapping(self):
        class D:
            pass

        class A:
            def __init__(self, d: D) -> None:
                self.d = d

        class BSuper:
            pass

        class BSub(BSuper):
            def __init__(self, a_map: Mapping[str, A]) -> None:
                self.a_map = a_map

        class C:
            def fn(self) -> D:
                return D()

        with mock_module(D, A, BSuper, BSub, C) as module:
            config = {
                "b": {
                    "class_path": f"{module}.BSub",
                    "init_args": {
                        "a_map": {
                            "name": {
                                "class_path": f"{module}.A",
                            },
                        },
                    },
                },
                "c": {},
            }
            config_path = os.path.join(self.tmpdir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f)

            parser = ArgumentParser()
            parser.add_argument("--config", action=ActionConfigFile)
            parser.add_subclass_arguments(BSuper, nested_key="b", required=True)
            parser.add_class_arguments(C, nested_key="c")
            parser.link_arguments("c", "b.init_args.a_map.name.init_args.d", compute_fn=C.fn, apply_on="instantiate")

            config = parser.parse_args(["--config", config_path])
            config_init = parser.instantiate_classes(config)
            self.assertIsInstance(config_init["b"].a_map["name"].d, D)

            config_init = parser.instantiate_classes(config)
            self.assertIsInstance(config_init["b"].a_map["name"].d, D)


    def test_subcommand_with_subclass_default_override_lightning_issue_10859(self):

        class Arch:
            def __init__(self, a: int = 1):
                pass

        class ArchB(Arch):
            def __init__(self, a: int = 2, b: int = 3):
                pass

        class ArchC(Arch):
            def __init__(self, a: int = 4, c: int = 5):
                pass

        parser = ArgumentParser(error_handler=None)
        parser_subcommands = parser.add_subcommands()
        subparser = ArgumentParser()
        subparser.add_argument('--arch', type=Arch)

        with mock_module(Arch, ArchB, ArchC) as module:
            default = {'class_path': f'{module}.ArchB'}
            value = {'class_path': f'{module}.ArchC', 'init_args': {'a': 10, 'c': 11}}

            subparser.set_defaults(arch=default)
            parser_subcommands.add_subcommand('fit', subparser)

            cfg = parser.parse_args(['fit', f'--arch={json.dumps(value)}'])
            self.assertEqual(cfg.fit.arch.as_dict(), value)


class OtherTests(unittest.TestCase):

    def test_is_optional(self):
        class MyEnum(Enum):
            A = 1

        params = [
            (Optional[bool],             bool, True),
            (Union[type(None), bool],    bool, True),
            (Dict[bool, type(None)],     bool, False),
            (Optional[Path_fr],          Path, True),
            (Union[type(None), Path_fr], Path, True),
            (Dict[Path_fr, type(None)],  Path, False),
            (Optional[MyEnum],           Enum, True),
            (Union[type(None), MyEnum],  Enum, True),
            (Dict[MyEnum, type(None)],   Enum, False),
        ]

        for typehint, ref_type, expected in params:
            with self.subTest(str(typehint)):
                self.assertEqual(expected, is_optional(typehint, ref_type))


if __name__ == '__main__':
    unittest.main(verbosity=2)
