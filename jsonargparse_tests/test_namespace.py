#!/usr/bin/env python3

import argparse
import unittest
from jsonargparse.namespace import Namespace, namespace_to_dict, dict_to_namespace
from jsonargparse_tests.base import is_cpython


class NamespaceTests(unittest.TestCase):

    def test_shallow_dot_set_get(self):
        ns = Namespace()
        ns.a = 1
        self.assertEqual(1, ns.a)
        self.assertEqual(ns, Namespace(a=1))

    def test_shallow_attr_set_get_del(self):
        ns = Namespace()
        setattr(ns, 'a', 1)
        self.assertEqual(1, getattr(ns, 'a'))
        self.assertEqual(ns, Namespace(a=1))
        delattr(ns, 'a')
        self.assertRaises(AttributeError, lambda: getattr(ns, 'a'))

    def test_shallow_item_set_get_del(self):
        ns = Namespace()
        ns['a'] = 1
        self.assertEqual(1, ns['a'])
        self.assertEqual(ns, Namespace(a=1))
        del ns['a']
        self.assertRaises(KeyError, lambda: ns['a'])

    def test_nested_item_set_get(self):
        ns = Namespace()
        ns['x.y.z'] = 1
        self.assertEqual(Namespace(x=Namespace(y=Namespace(z=1))), ns)
        self.assertEqual(1, ns['x.y.z'])
        self.assertEqual(1, ns['x']['y']['z'])
        self.assertEqual(Namespace(z=1), ns['x.y'])
        self.assertEqual(Namespace(z=1), ns['x']['y'])
        ns['x.y'] = 2
        self.assertEqual(2, ns['x.y'])

    def test_nested_item_set_del(self):
        ns = Namespace()
        ns['x.y'] = 1
        self.assertEqual(Namespace(x=Namespace(y=1)), ns)
        del ns['x.y']
        self.assertEqual(Namespace(x=Namespace()), ns)

    def test_get(self):
        ns = Namespace()
        ns['x.y'] = 1
        self.assertEqual(1, ns.get('x.y'))
        self.assertEqual(Namespace(y=1), ns.get('x'))
        self.assertEqual(2, ns.get('z', 2))
        self.assertIsNone(ns.get('z'))

    def test_get_non_str_key(self):
        ns = Namespace()
        for key in [None, True, False, 1, 2.3]:
            with self.subTest(str(key)):
                self.assertIsNone(ns.get(key))
                self.assertEqual(ns.get(key, 'abc'), 'abc')

    def test_set_item_nested_dict(self):
        ns = Namespace(d={'a': 1})
        ns['d.b'] = 2
        self.assertEqual(2, ns['d']['b'])

    def test_contains_non_str_key(self):
        ns = Namespace()
        for key in [None, True, False, 1, 2.3]:
            with self.subTest(str(key)):
                self.assertNotIn(key, ns)

    def test_pop(self):
        ns = Namespace()
        ns['x.y.z'] = 1
        self.assertEqual(1, ns.pop('x.y.z'))
        self.assertEqual(ns, Namespace(x=Namespace(y=Namespace())))

    def test_nested_item_invalid_set(self):
        ns = Namespace()
        with self.assertRaises(KeyError):
            ns['x.'] = 1
        with self.assertRaises(KeyError):
            ns['x .y'] = 2

    def test_nested_key_in(self):
        ns = Namespace()
        ns['x.y.z'] = 1
        self.assertIn('x', ns)
        self.assertIn('x.y', ns)
        self.assertIn('x.y.z', ns)
        self.assertNotIn('a', ns)
        self.assertNotIn('x.a', ns)
        self.assertNotIn('x.y.a', ns)
        self.assertNotIn('x.y.z.a', ns)
        self.assertNotIn('x..y', ns)
        self.assertNotIn(123, ns)

    @unittest.skipIf(not is_cpython, 'requires dict insertion order')
    def test_items_generator(self):
        ns = Namespace()
        ns['a'] = 1
        ns['b.c'] = 2
        ns['b.d'] = 3
        ns['p.q.r'] = {'x': 4, 'y': 5}
        items = list(ns.items())
        self.assertEqual(items, [('a', 1), ('b.c', 2), ('b.d', 3), ('p.q.r', {'x': 4, 'y': 5})])

    @unittest.skipIf(not is_cpython, 'requires dict insertion order')
    def test_keys_generator(self):
        ns = Namespace()
        ns['a'] = 1
        ns['b.c'] = 2
        ns['b.d'] = 3
        ns['p.q.r'] = {'x': 4, 'y': 5}
        keys = list(ns.keys())
        self.assertEqual(keys, ['a', 'b.c', 'b.d', 'p.q.r'])

    @unittest.skipIf(not is_cpython, 'requires dict insertion order')
    def test_values_generator(self):
        ns = Namespace()
        ns['a'] = 1
        ns['b.c'] = 2
        ns['b.d'] = 3
        ns['p.q.r'] = {'x': 4, 'y': 5}
        values = list(ns.values())
        self.assertEqual(values, [1, 2, 3, {'x': 4, 'y': 5}])

    def test_namespace_from_dict(self):
        dic = {'a': 1, 'b': {'c': 2}}
        ns = Namespace(dic)
        self.assertEqual(ns, Namespace(a=1, b={'c': 2}))

    def test_as_dict(self):
        ns = Namespace()
        ns['w'] = 1
        ns['x.y'] = 2
        ns['x.z'] = 3
        ns['p'] = {'q': Namespace(r=4)}
        self.assertEqual(ns.as_dict(), {'w': 1, 'x': {'y': 2, 'z': 3}, 'p': {'q': {'r': 4}}})
        self.assertEqual(Namespace().as_dict(), {})

    def test_as_flat(self):
        ns = Namespace()
        ns['w'] = 1
        ns['x.y.z'] = 2
        flat = ns.as_flat()
        self.assertIsInstance(flat, argparse.Namespace)
        self.assertEqual(vars(flat), {'w': 1, 'x.y.z': 2})

    def test_clone(self):
        ns = Namespace()
        pqr = {'x': 4, 'y': 5}
        ns['a'] = 1
        ns['p.q.r'] = pqr
        self.assertIs(ns['p.q.r'], pqr)
        self.assertEqual(ns.clone(), ns)
        self.assertIsNot(ns.clone()['p.q.r'], pqr)
        self.assertIsNot(ns.clone()['p.q'], ns['p.q'])

    def test_update_shallow(self):
        ns_from = Namespace(a=1, b=None)
        ns_to = Namespace(a=None, b=2, c=3)
        ns_to.update(ns_from)
        self.assertEqual(ns_to, Namespace(a=1, b=None, c=3))

    def test_update_invalid(self):
        ns = Namespace()
        self.assertRaises(KeyError, lambda: ns.update(123))

    def test_init_from_argparse_flat_namespace(self):
        argparse_ns = argparse.Namespace()
        setattr(argparse_ns, 'w', 0)
        setattr(argparse_ns, 'x.y.a', 1)
        setattr(argparse_ns, 'x.y.b', 2)
        setattr(argparse_ns, 'z.c', 3)
        ns = Namespace(argparse_ns)
        self.assertEqual(ns, Namespace(w=0, x=Namespace(y=Namespace(a=1, b=2)), z=Namespace(c=3)))

    def test_init_invalid(self):
        self.assertRaises(ValueError, lambda: Namespace(1))
        self.assertRaises(ValueError, lambda: Namespace(argparse.Namespace(), x=1))

    def test_namespace_to_dict(self):
        ns = Namespace()
        ns['w'] = 1
        ns['x.y'] = 2
        ns['x.z'] = 3
        dic1 = namespace_to_dict(ns)
        dic2 = ns.as_dict()
        self.assertEqual(dic1, dic2)
        self.assertIsNot(dic1, dic2)

    def test_dict_to_namespace(self):
        ns1 = Namespace(a=1, b=Namespace(c=2), d=[Namespace(e=3)])
        dic = {'a': 1, 'b': {'c': 2}, 'd': [{'e': 3}]}
        ns2 = dict_to_namespace(dic)
        self.assertEqual(ns1, ns2)

    def test_use_for_kwargs(self):
        def func(a=1, b=2, c=3):
            return a, b, c

        kwargs = Namespace(a=4, c=5)
        val = func(**kwargs)
        self.assertEqual(val, (4, 2, 5))


if __name__ == '__main__':
    unittest.main(verbosity=2)
