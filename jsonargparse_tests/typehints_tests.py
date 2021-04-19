#!/usr/bin/env python3
# pylint: disable=unsubscriptable-object

import json
import pathlib
import sys
import uuid
import yaml
from calendar import Calendar
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from jsonargparse_tests.base import *
from jsonargparse.typehints import is_optional, Literal


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


    def test_list_path(self):
        parser = ArgumentParser()
        parser.add_argument('--paths', type=List[Path_fc])
        cfg = parser.parse_args(['--paths=["file1", "file2"]'])
        self.assertEqual(['file1', 'file2'], cfg.paths)
        self.assertIsInstance(cfg.paths[0], Path)
        self.assertIsInstance(cfg.paths[1], Path)


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
        parser = ArgumentParser(error_handler=None, parse_as_dict=True)
        parser.add_argument('--dict', type=dict)
        self.assertEqual({}, parser.parse_args(['--dict={}'])['dict'])
        self.assertEqual({'a': 1, 'b': '2'}, parser.parse_args(['--dict={"a":1, "b":"2"}'])['dict'])
        self.assertRaises(ParserError, lambda: parser.parse_args(['--dict=1']))


    def test_dict_union(self):
        class MyEnum(Enum):
            ab = 1

        parser = ArgumentParser(error_handler=None, parse_as_dict=True)
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
        parser = ArgumentParser(error_handler=None, parse_as_dict=True)
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


    @unittest.skipIf(sys.version_info.minor < 8, 'Literal introduced in python 3.8')
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
        parser = ArgumentParser(error_handler=None, parse_as_dict=True)
        parser.add_argument('--op', type=Optional[List[Calendar]])

        class_path = '"class_path": "calendar.Calendar"'
        cfg = parser.parse_args(['--op=[{'+class_path+'}]'])
        self.assertEqual(cfg['op'], [{'class_path': 'calendar.Calendar', "init_args": {"firstweekday": 0}}])
        cfg = parser.parse_args(['--op=["calendar.Calendar"]'])
        self.assertEqual(cfg['op'], [{'class_path': 'calendar.Calendar', "init_args": {"firstweekday": 0}}])
        cfg = parser.instantiate_subclasses(cfg)
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
        self.assertEqual(cfg['op'][0]['init_args'], {'firstweekday': 3})
        cfg = parser.instantiate_subclasses(cfg)
        self.assertIsInstance(cfg['op'][0], Calendar)
        self.assertEqual(3, cfg['op'][0].firstweekday)

        parser = ArgumentParser(parse_as_dict=True)
        parser.add_argument('--n.op', type=Optional[Calendar])
        cfg = parser.parse_args(['--n.op={'+class_path+', '+init_args+'}'])
        cfg = parser.instantiate_subclasses(cfg)
        self.assertIsInstance(cfg['n']['op'], Calendar)
        self.assertEqual(3, cfg['n']['op'].firstweekday)

        parser = ArgumentParser()
        parser.add_argument('--op', type=Calendar)
        cfg = parser.parse_args(['--op={'+class_path+', '+init_args+'}'])
        cfg = parser.instantiate_subclasses(cfg)
        self.assertIsInstance(cfg['op'], Calendar)
        self.assertEqual(3, cfg['op'].firstweekday)

        cfg = parser.instantiate_subclasses(parser.parse_args([]))
        self.assertIsNone(cfg['op'])


    def test_typehint_serialize_list(self):
        val = [PositiveInt(1), PositiveInt(2)]
        self.assertEqual([1, 2], ActionTypeHint.serialize(val, Union[PositiveInt, List[PositiveInt]]))
        self.assertRaises(ValueError, lambda: ActionTypeHint.serialize([1, -2], List[PositiveInt]))  # type: ignore


    def test_typehint_serialize_enum(self):

        class MyEnum(Enum):
            a = 1
            b = 2

        self.assertEqual('b', ActionTypeHint.serialize(MyEnum.b, Optional[MyEnum]))
        self.assertRaises(ValueError, lambda: ActionTypeHint.serialize('x', Optional[MyEnum]))


    def test_unsupported_type(self):
        self.assertRaises(ValueError, lambda: ActionTypeHint(typehint=lambda: None))
        self.assertRaises(ValueError, lambda: ActionTypeHint(typehint=Union[int, lambda: None]))


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


class TypeHintsTmpdirTests(TempDirTestCase):

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

        parser = ArgumentParser(parse_as_dict=True)
        parser.add_argument('--data', type=Dict[str, Any], enable_path=True)
        parser.add_argument('--cal', type=Calendar, enable_path=True)
        cfg = parser.parse_args(['--data=data.yaml'])
        self.assertEqual('data.yaml', str(cfg['data'].pop('__path__')))
        self.assertEqual(data, cfg['data'])
        cfg = parser.instantiate_subclasses(parser.parse_args(['--cal=cal.yaml']))
        self.assertIsInstance(cfg['cal'], Calendar)


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

        cfg = namespace_to_dict(parser.get_defaults())
        self.assertEqual(config_path, str(cfg['__default_config__']))
        self.assertEqual(cfg['data']['cal'], config)
        cfg = parser.dump(cfg)
        self.assertIn('class_path: calendar.Calendar\n', cfg)
        self.assertIn('firstweekday: 3\n', cfg)


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
