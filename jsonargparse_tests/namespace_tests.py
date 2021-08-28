#!/usr/bin/env python3

import unittest
from jsonargparse.namespace import ArgparseNamespace, Namespace


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

    def test_nested_item_invalid_set(self):
        ns = Namespace()
        with self.assertRaises(KeyError):
            ns['x.'] = 1
        with self.assertRaises(KeyError):
            ns['x .y'] = 2
        ns['x.y'] = 3
        with self.assertRaises(KeyError):
            ns['x.y.z'] = 4

    def test_nested_key_in(self):
        ns = Namespace()
        ns['x.y.z'] = 1
        self.assertTrue('x' in ns)
        self.assertTrue('x.y' in ns)
        self.assertTrue('x.y.z' in ns)
        self.assertFalse('a' in ns)
        self.assertFalse('x.a' in ns)
        self.assertFalse('x.y.a' in ns)
        self.assertFalse('x.y.z.a' in ns)

    def test_items_generator(self):
        ns = Namespace()
        ns['a'] = 1
        ns['b.c'] = 2
        ns['b.d'] = 3
        ns['p.q.r'] = {'x': 4, 'y': 5}
        items = list(ns.items())
        self.assertEqual(items, [('a', 1), ('b.c', 2), ('b.d', 3), ('p.q.r', {'x': 4, 'y': 5})])

    def test_keys_generator(self):
        ns = Namespace()
        ns['a'] = 1
        ns['b.c'] = 2
        ns['b.d'] = 3
        ns['p.q.r'] = {'x': 4, 'y': 5}
        keys = list(ns.keys())
        self.assertEqual(keys, ['a', 'b.c', 'b.d', 'p.q.r'])

    def test_values_generator(self):
        ns = Namespace()
        ns['a'] = 1
        ns['b.c'] = 2
        ns['b.d'] = 3
        ns['p.q.r'] = {'x': 4, 'y': 5}
        values = list(ns.values())
        self.assertEqual(values, [1, 2, 3, {'x': 4, 'y': 5}])

    def test_as_dict(self):
        ns = Namespace()
        ns['w'] = 1
        ns['x.y'] = 2
        ns['x.z'] = 3
        self.assertEqual(ns.as_dict(), {'w': 1, 'x': {'y': 2, 'z': 3}})
        self.assertEqual(Namespace().as_dict(), {})

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

    def test_init_from_argparse_flat_namespace(self):
        argparse_ns = ArgparseNamespace()
        setattr(argparse_ns, 'w', 0)
        setattr(argparse_ns, 'x.y.a', 1)
        setattr(argparse_ns, 'x.y.b', 2)
        setattr(argparse_ns, 'z.c', 3)
        ns = Namespace(argparse_ns)
        self.assertEqual(ns, Namespace(w=0, x=Namespace(y=Namespace(a=1, b=2)), z=Namespace(c=3)))

    def test_init_invalid(self):
        self.assertRaises(ValueError, lambda: Namespace(1))
        self.assertRaises(ValueError, lambda: Namespace(ArgparseNamespace(), x=1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
