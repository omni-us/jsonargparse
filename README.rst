.. image:: https://circleci.com/gh/omni-us/jsonargparse.svg?style=svg
    :target: https://circleci.com/gh/omni-us/jsonargparse
.. image:: https://codecov.io/gh/omni-us/jsonargparse/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/omni-us/jsonargparse
.. image:: https://sonarcloud.io/api/project_badges/measure?project=omni-us_jsonargparse&metric=alert_status
    :target: https://sonarcloud.io/dashboard?id=omni-us_jsonargparse
.. image:: https://badge.fury.io/py/jsonargparse.svg
    :target: https://badge.fury.io/py/jsonargparse
.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg
    :target: https://github.com/omni-us/jsonargparse


jsonargparse
============

https://jsonargparse.readthedocs.io/en/stable/

This package is an extension to python's argparse which simplifies parsing of
configuration options from command line arguments, json configuration files
(`yaml <https://yaml.org/>`__ or `jsonnet <https://jsonnet.org/>`__ supersets),
environment variables and hard-coded defaults.

The aim is similar to other projects such as `configargparse
<https://pypi.org/project/ConfigArgParse/>`__, `yconf
<https://pypi.org/project/yconf/>`__, `confuse
<https://pypi.org/project/confuse/>`__, `typer
<https://pypi.org/project/typer/>`__, `OmegaConf
<https://pypi.org/project/omegaconf/>`__, `Fire
<https://pypi.org/project/fire/>`__ and `click
<https://pypi.org/project/click/>`__. The obvious question is, why yet another
package similar to many already existing ones? The answer is simply that none of
the existing projects had the exact features we wanted and after analyzing the
alternatives it seemed simpler to start a new project.


Features
========

- Great support of type hint annotations for automatic creation of parsers and
  minimal boilerplate command line interfaces.

- Not exclusively intended for parsing command line arguments. The main focus is
  parsing configuration files and not necessarily from a command line tool.

- Support for nested namespaces which makes it possible to parse config files
  with non-flat hierarchies.

- Parsing of relative paths within config files and path lists.

- Support for two popular supersets of json: yaml and jsonnet.

- Parsers can be configured just like with python's argparse, thus it has a
  gentle learning curve.

- Several convenient types and action classes to ease common parsing use cases
  (paths, comparison operators, json schemas, enums ...).

- Support for command line tab argument completion using `argcomplete
  <https://pypi.org/project/argcomplete/>`__.

- Configuration values are overridden based on the following precedence.

  - **Parsing command line:** command line arguments (might include config file)
    > environment variables > default config file > defaults.
  - **Parsing files:** config file > environment variables > default config file
    > defaults.
  - **Parsing environment:** environment variables > default config file >
    defaults.


.. _installation:

Installation
============

You can install using `pip <https://pypi.org/project/jsonargparse/>`__ as:

.. code-block:: bash

    pip install jsonargparse

By default the only dependency that jsonargparse installs is `PyYAML
<https://pypi.org/project/PyYAML/>`__. However, jsonargparse has several
optional features that can be enabled by specifying any of the following extras
requires: :code:`signatures`, :code:`jsonschema`, :code:`jsonnet`, :code:`urls`,
:code:`argcomplete` and :code:`reconplogger`. There is also the :code:`all`
extras require that can be used to enable all optional features. Installing
jsonargparse with extras require is as follows:

.. code-block:: bash

    pip install "jsonargparse[signatures,urls]"  # Enable signatures and URLs features
    pip install "jsonargparse[all]"              # Enable all optional features

The following table references sections that describe optional features and the
corresponding extras requires that enables them.

+----------------------------------+-------------+-------------+---------+------------+------------+
|                                  | urls/fsspec | argcomplete | jsonnet | jsonschema | signatures |
+----------------------------------+-------------+-------------+---------+------------+------------+
| :ref:`type-hints`                |             |             |         |            | ✓          |
+----------------------------------+-------------+-------------+---------+------------+------------+
| :ref:`classes-methods-functions` |             |             |         |            | ✓          |
+----------------------------------+-------------+-------------+---------+------------+------------+
| :ref:`sub-classes`               |             |             |         |            | ✓          |
+----------------------------------+-------------+-------------+---------+------------+------------+
| :ref:`parsing-urls`              | ✓           |             |         |            |            |
+----------------------------------+-------------+-------------+---------+------------+------------+
| :ref:`json-schemas`              |             |             |         | ✓          |            |
+----------------------------------+-------------+-------------+---------+------------+------------+
| :ref:`jsonnet-files`             |             |             | ✓       |            |            |
+----------------------------------+-------------+-------------+---------+------------+------------+
| :ref:`tab-completion`            |             | ✓           |         |            |            |
+----------------------------------+-------------+-------------+---------+------------+------------+


Basic usage
===========

There are multiple ways of using jsonargparse. The most simple way which
requires to write the least amount of code is by using the :func:`.CLI`
function, for example:

.. code-block:: python

    from jsonargparse import CLI

    def command(
        name: str,
        prize: int = 100
    ):
        """
        Args:
            name: Name of winner.
            prize: Amount won.
        """
        print(f'{name} won {prize}€!')

    if __name__ == '__main__':
        CLI()

Then in a shell you could run:

.. code-block:: bash

    $ python example.py Lucky --prize=1000
    Lucky won 1000€!

:func:`.CLI` without arguments searches for functions and classes defined in the
same module and in the local context where :func:`.CLI` is called. Giving a
single or a list functions/classes as first argument to :func:`.CLI` skips the
automatic search and only includes what is given.

When :func:`.CLI` receives a single class, the first arguments are used to
instantiate the class, then a class method name must be given (i.e. methods
become :ref:`sub-commands`) and the remaining arguments are used to run the
class method. An example would be:

.. code-block:: python

    from random import randint
    from jsonargparse import CLI

    class Main:
        def __init__(
            self,
            max_prize: int = 100
        ):
            """
            Args:
                max_prize: Maximum prize that can be awarded.
            """
            self.max_prize = max_prize

        def person(
            self,
            name: str
        ):
            """
            Args:
                name: Name of winner.
            """
            return f'{name} won {randint(0, self.max_prize)}€!'

    if __name__ == '__main__':
        print(CLI(Main))

Then in a shell you could run:

.. code-block:: bash

    $ python example.py --max_prize=1000 person Lucky
    Lucky won 632€!

If more than one function is given to :func:`.CLI`, then any of them can be
executed via :ref:`sub-commands` similar to the single class example above, i.e.
:code:`example.py function [arguments]` where :code:`function` is the name of
the function to execute.

If multiple classes or a mixture of functions and classes is given to
:func:`.CLI`, to execute a method of a class, two levels of :ref:`sub-commands`
are required. The first sub-command would be the name of the class and the
second the name of the method, i.e. :code:`example.py class [init_arguments]
method [arguments]`. For more details about the automatic adding of arguments
from classes and functions and the use of configuration files refer to section
:ref:`classes-methods-functions`.

This simple way of usage is similar and inspired by `Fire
<https://pypi.org/project/fire/>`__. However, there are fundamental differences.
First, the purpose is not allowing to call any python object from the command
line. It is only intended for running functions and classes specifically written
for this purpose. Second, the arguments are required to have type hints, and the
values will be validated according to these. Third, the return values of the
functions are not automatically printed. :func:`.CLI` returns its value and it
is up to the developer to decide what to do with it. Finally, jsonargparse has
many features designed to help in creating convenient argument parsers such as:
:ref:`nested-namespaces`, :ref:`configuration-files`, additional type hints
(:ref:`parsing-paths`, :ref:`restricted-numbers`, :ref:`restricted-strings`) and
much more.

The next section explains how to create an argument parser in a low level
argparse-style. However, as parsers get more complex, being able to define them
in a modular way becomes important. Three mechanisms are available to define
parsers in a modular way, see respective sections
:ref:`classes-methods-functions`, :ref:`sub-commands` and
:ref:`parser-arguments`.


Parsers
=======

An argument parser is created just like it is done with python's `argparse
<https://docs.python.org/3/library/argparse.html>`__. You import the module,
create a parser object and then add arguments to it. A simple example would be:

.. code-block:: python

    from jsonargparse import ArgumentParser
    parser = ArgumentParser(
        prog='app',
        description='Description for my app.')

    parser.add_argument('--opt1',
        type=int,
        default=0,
        help='Help for option 1.')

    parser.add_argument('--opt2',
        type=float,
        default=1.0,
        help='Help for option 2.')


After creating the parser, you can use it to parse command line arguments with
the :py:meth:`.ArgumentParser.parse_args` function, after which you get
an object with the parsed values or defaults available as attributes. For
illustrative purposes giving to :func:`parse_args` a list of arguments (instead
of automatically getting them from the command line arguments), with the parser
shown above you would observe:

.. code-block:: python

    >>> cfg = parser.parse_args(['--opt2', '2.3'])
    >>> cfg.opt1, type(cfg.opt1)
    (0, <class 'int'>)
    >>> cfg.opt2, type(cfg.opt2)
    (2.3, <class 'float'>)

If the parsing fails the standard behavior is that the usage is printed and the
program is terminated. Alternatively you can initialize the parser with
:code:`error_handler=None` in which case a :class:`.ParserError` is raised.


.. _nested-namespaces:

Nested namespaces
=================

A difference with respect to the basic argparse is that it by using dot notation
in the argument names, you can define a hierarchy of nested namespaces. So for
example you could do the following:

.. code-block:: python

    >>> parser = ArgumentParser(prog='app')
    >>> parser.add_argument('--lev1.opt1', default='from default 1')
    >>> parser.add_argument('--lev1.opt2', default='from default 2')
    >>> cfg = parser.get_defaults()
    >>> cfg.lev1.opt1
    'from default 2'
    >>> cfg.lev1.opt2
    'from default 2'

A group of nested options can be created by using a dataclass. This has the
advantage that the same options can be reused in multiple places of a project.
An example analogous to the one above would be:

.. code-block:: python

    from dataclasses import dataclass

    @dataclass
    class Level1Options:
        """Level 1 options
        Args:
            opt1: Option 1
            opt2: Option 2
        """
        opt1: str = 'from default 1'
        opt2: str = 'from default 2'

    parser = ArgumentParser()
    parser.add_argument('--lev1', type=Level1Options, default=Level1Options())


.. _configuration-files:

Configuration files
===================

An important feature of jsonargparse is the parsing of yaml/json files. The dot
notation hierarchy of the arguments (see :ref:`nested-namespaces`) are used for
the expected structure in the config files.

The :py:attr:`.ArgumentParser.default_config_files` property can be set when
creating a parser to specify patterns to search for configuration files. For
example if a parser is created as
:code:`ArgumentParser(default_config_files=['~/.myapp.yaml',
'/etc/myapp.yaml'])`, when parsing if any of those two config files exist it
will be parsed and used to override the defaults. Only the first matched config
file is used. The default config file is always parsed first, this means that
any command line arguments will override its values.

It is also possible to add an argument to explicitly provide a configuration
file path. Providing a config file as an argument does not disable the parsing
of :code:`default_config_files`. The config argument would be parsed in the
specific position among the command line arguments. Therefore the arguments
found after would override the values from the config file. The config argument
can be given multiple times, each overriding the values of the previous. Using
the example parser from the :ref:`nested-namespaces` section above, we could
have the following config file in yaml format:

.. code-block:: yaml

    # File: example.yaml
    lev1:
      opt1: from yaml 1
      opt2: from yaml 2

Then in python adding a config file argument and parsing some dummy arguments,
the following would be observed:

.. code-block:: python

    >>> from jsonargparse import ArgumentParser, ActionConfigFile
    >>> parser = ArgumentParser()
    >>> parser.add_argument('--lev1.opt1', default='from default 1')
    >>> parser.add_argument('--lev1.opt2', default='from default 2')
    >>> parser.add_argument('--cfg', action=ActionConfigFile)
    >>> cfg = parser.parse_args(['--lev1.opt1', 'from arg 1',
                                 '--cfg', 'example.yaml',
                                 '--lev1.opt2', 'from arg 2'])
    >>> cfg.lev1.opt1
    'from yaml 1'
    >>> cfg.lev1.opt2
    'from arg 2'

Instead of providing a path to a configuration file, a string with the
configuration content can also be provided.

.. code-block:: python

    >>> cfg = parser.parse_args(['--cfg', '{"lev1":{"opt1":"from string 1"}}'])
    >>> cfg.lev1.opt1
    'from string 1'

The config file can also be provided as an environment variable as explained in
section :ref:`environment-variables`. The configuration file environment
variable is the first one to be parsed. So any argument provided through an
environment variable would override the config file one.

A configuration file or string can also be parsed without parsing command line
arguments. The methods for this are :py:meth:`.ArgumentParser.parse_path` and
:py:meth:`.ArgumentParser.parse_string` to parse a config file or a config
string respectively.

Parsers that have an :class:`.ActionConfigFile` also include a
:code:`--print_config` option. This is useful particularly for command line
tools with a large set of options to create an initial config file including all
default values. If the `ruyaml <https://ruyaml.readthedocs.io>`__ package is
installed, the config can be printed having the help descriptions content as
yaml comments by using :code:`--print_config=comments`. Another option is
:code:`--print_config=skip_null` which skips entries whose value is
:code:`null`.


.. _environment-variables:

Environment variables
=====================

The jsonargparse parsers can also get values from environment variables. The
parser checks existing environment variables whose name is of the form
:code:`[PREFIX_][LEV__]*OPT`, that is, all in upper case, first a prefix (set by
:code:`env_prefix`, or if unset the :code:`prog` without extension) followed by
underscore and then the argument name replacing dots with two underscores. Using
the parser from the :ref:`nested-namespaces` section above, in your shell you
would set the environment variables as:

.. code-block:: bash

    export APP_LEV1__OPT1='from env 1'
    export APP_LEV1__OPT2='from env 2'

Then in python the parser would use these variables, unless overridden by the
command line arguments, that is:

.. code-block:: python

    >>> parser = ArgumentParser(env_prefix='APP', default_env=True)
    >>> parser.add_argument('--lev1.opt1', default='from default 1')
    >>> parser.add_argument('--lev1.opt2', default='from default 2')
    >>> cfg = parser.parse_args(['--lev1.opt1', 'from arg 1'])
    >>> cfg.lev1.opt1
    'from arg 1'
    >>> cfg.lev1.opt2
    'from env 2'

Note that when creating the parser, :code:`default_env=True` was given. By
default :py:meth:`.ArgumentParser.parse_args` does not check environment
variables, so it has to be enabled explicitly.

There is also the :py:meth:`.ArgumentParser.parse_env` function to only parse
environment variables, which might be useful for some use cases in which there
is no command line call involved.

If a parser includes an :class:`.ActionConfigFile` argument, then the
environment variable for this config file will be checked before all the other
environment variables.


.. _classes-methods-functions:

Classes, methods and functions
==============================

It is good practice to write python code in which parameters have type hints and
are described in the docstrings. To make this well written code configurable, it
wouldn't make sense to duplicate information of types and parameter
descriptions. To avoid this duplication, jsonargparse includes methods to
automatically add their arguments:
:py:meth:`.SignatureArguments.add_class_arguments`,
:py:meth:`.SignatureArguments.add_method_arguments`,
:py:meth:`.SignatureArguments.add_function_arguments` and
:py:meth:`.SignatureArguments.add_dataclass_arguments`.

Take for example a class with its init and a method with docstrings as follows:

.. code-block:: python

    from typing import Dict, Union, List

    class MyClass(MyBaseClass):
        def __init__(self, items: Dict[str, Union[int, List[int]]], **kwargs):
            """Initializer for MyClass.

            Args:
                items: Description for items.
            """
            pass

        def mymethod(self, value: float, flag: bool = False):
            """Description for mymethod.

            Args:
                value: Description for value.
                flag: Description for flag.
            """
            pass

Both :code:`MyClass` and :code:`mymethod` can easily be made configurable, the
class initialized and the method executed as follows:

.. code-block:: python

    from jsonargparse import ArgumentParser, namespace_to_dict

    parser = ArgumentParser()
    parser.add_class_arguments(MyClass, 'myclass.init')
    parser.add_method_arguments(MyClass, 'mymethod', 'myclass.method')

    cfg = parser.parse_args()
    myclass = MyClass(**namespace_to_dict(cfg.myclass.init))
    myclass.mymethod(**namespace_to_dict(cfg.myclass.method))

The :func:`add_class_arguments` call adds to the *myclass.init* key the
:code:`items` argument with description as in the docstring, it is set as
required since it does not have a default value, and when parsed it is validated
according to its type hint, i.e., a dict with values ints or list of ints. Also
since the init has the :code:`**kwargs` argument, the keyword arguments from
:code:`MyBaseClass` are also added to the parser. Similarly the
:func:`add_method_arguments` call adds to the *myclass.method* key the arguments
:code:`value` as a required float and :code:`flag` as an optional boolean with
default value false.

Instead of using :func:`namespace_to_dict` to convert the namespaces to a
dictionary, the :class:`.ArgumentParser` object can be instantiated with
:code:`parse_as_dict=True` to get directly a dictionary from the parsing
methods.

When parsing from a configuration file (see :ref:`configuration-files`) all the
values can be given in a single config file. However, for convenience it is also
possible that the values for each of the groups created by the calls to the add
signature methods can be parsed from independent files. This means that for the
example above there could be one general config file with contents:

.. code-block:: yaml

    myclass:
      init: myclass.yaml
      method: mymethod.yaml

Then the files :code:`myclass.yaml` and :code:`mymethod.yaml` would only include
the settings for each of the instantiation of the class and the call to the
method respectively.

A wide range of type hints are supported. For exact details go to section
:ref:`type-hints`. Some notes about the support for automatic adding of
arguments are:

- All positional arguments must have a type, otherwise the add arguments
  functions raise an exception.

- Keyword arguments are ignored if they don't have at least one type that is
  supported. They are also ignored if the name starts with :code:`_`.

- Recursive adding of arguments from base classes only considers the presence
  of :code:`*args` and :code:`**kwargs`. It does not check the code to identify
  if :code:`super().__init__` is called or with which arguments.

- Arguments whose name starts with :code:`_` are considered for internal use
  and ignored.

- The signature methods have a :code:`skip` parameters which can be used to
  exclude adding some arguments, e.g.
  :code:`parser.add_method_arguments(MyClass, 'mymethod', skip={'flag'})`.

.. note::

    Since keyword arguments with unsupported types are ignored, during
    development it might be desired to know which arguments are ignored and the
    specific reason. This can be done by initializing :class:`.ArgumentParser`
    with :code:`logger={'level': 'DEBUG'}`. For more details about logging go to
    section :ref:`logging`.

.. note::

    For all features described above to work, one optional package is required:
    `docstring-parser <https://pypi.org/project/docstring-parser/>`__ to get the
    argument descriptions from the docstrings. This package is included when
    jsonargparse is installed using the :code:`signatures` extras require as
    explained in section :ref:`installation`.


.. _argument-linking:

Argument linking
================

Some use cases could require adding arguments from multiple classes and be
desired that some parameters get a value automatically computed from other
arguments. This behavior can be obtained by using the
:py:meth:`.ArgumentParser.link_arguments` method.

There are two types of links each defined with :code:`apply_on='parse'` and
:code:`apply_on='instantiate'`. As the names suggest the former are set when
calling one of the parse methods and the latter are set when calling
:py:meth:`.ArgumentParser.instantiate_classes`.

For parsing links, source keys can be individual arguments or nested groups. The
target key has to be a single argument. The keys can be inside init_args of a
subclass. The compute function should accept as many positional arguments as
there are sources and return a value of type compatible with the target. An
example would be the following:

.. code-block:: python

    class Model:
        def __init__(self, batch_size: int):
            self.batch_size = batch_size

    class Data:
        def __init__(self, batch_size: int = 5):
            self.batch_size = batch_size

    parser = ArgumentParser()
    parser.add_class_arguments(Model, 'model')
    parser.add_class_arguments(Data, 'data')
    parser.link_arguments('data.batch_size', 'model.batch_size', apply_on='parse')

As argument and in config files only :code:`data.batch_size` should be
specified. Then whatever value it has will be propagated to
:code:`model.batch_size`.

For instantiation links, only a single source key is supported. The key can be
for a class group created using
:py:meth:`.SignatureArguments.add_class_arguments` or a subclass action created
using :py:meth:`.SignatureArguments.add_subclass_arguments`. If the key is only
the class group or subclass action, then a compute function is required which
takes the source class instance and returns the value to set in target.
Alternatively the key can specify a class attribute. The target key has to be a
single argument and can be inside init_args of a subclass. The order of
instantiation used by :py:meth:`.ArgumentParser.instantiate_classes` is
automatically determined based on the links. The instantiation links must be
a directed acyclic graph. An example would be the following:

.. code-block:: python

    class Model:
        def __init__(self, num_classes: int):
            self.num_classes = num_classes

    class Data:
        def __init__(self):
            self.num_classes = get_num_classes()

    parser = ArgumentParser()
    parser.add_class_arguments(Model, 'model')
    parser.add_class_arguments(Data, 'data')
    parser.link_arguments('data.num_classes', 'model.num_classes', apply_on='instantiate')

This link would imply that :py:meth:`.ArgumentParser.instantiate_classes`
instantiates :class:`Data` first, then use the :code:`num_classes` attribute to
instantiate :class:`Model`.


.. _type-hints:

Type hints
==========

As explained in section :ref:`classes-methods-functions` type hints are required
to automatically add arguments from signatures to a parser. Additional to this
feature, a type hint can also be used independently when adding a single
argument to the parser. For example, an argument that can be :code:`None` or a
float in the range :code:`(0, 1)` or a positive int could be added using a type
hint as follows:

.. code-block:: python

    from typing import Optional, Union
    from jsonargparse.typing import PositiveInt, OpenUnitInterval
    parser.add_argument('--op', type=Optional[Union[PositiveInt, OpenUnitInterval]])

The support of type hints is designed to not require developers to change their
types or default values. In other words, the idea is to support type hints
whatever they may be, as opposed to requiring jsonargparse specific types. The
types included in :code:`jsonargparse.typing` are completely generic and could
even be useful independent of the argument parsers.

A wide range of type hints are supported and with arbitrary complexity/nesting.
Some notes about this support are:

- Nested types are supported as long as at least one child type is supported.

- Fully supported types are: :code:`str`, :code:`bool`, :code:`int`,
  :code:`float`, :code:`complex`, :code:`List`, :code:`Iterable`,
  :code:`Sequence`, :code:`Any`, :code:`Union`, :code:`Optional`, :code:`Type`,
  :code:`Enum`, :code:`UUID`, restricted types as explained in sections
  :ref:`restricted-numbers` and :ref:`restricted-strings` and paths and URLs as
  explained in sections :ref:`parsing-paths` and :ref:`parsing-urls`.

- :code:`Dict` is supported but only with :code:`str` or :code:`int` keys.

- :code:`Tuple` and :code:`Set` are supported even though they can't be
  represented in json distinguishable from a list. Each :code:`Tuple` element
  position can have its own type and will be validated as such. :code:`Tuple`
  with ellipsis (:code:`Tuple[type, ...]`) is also supported. In command line
  arguments, config files and environment variables, tuples and sets are
  represented as an array.

- :code:`dataclasses` are supported as a type but without any nesting and for
  pure data classes. By pure it is meant that it only inherits from data
  classes, not a mixture of normal classes and data classes.

- To set a value to :code:`None` it is required to use :code:`null` since this
  is how json/yaml defines it. To avoid confusion in the help, :code:`NoneType`
  is displayed as :code:`null`. For example a function argument with type and
  default :code:`Optional[str] = None` would be shown in the help as
  :code:`type: Union[str, null], default: null`.

- :code:`Callable` has an experimental partial implementation and not officially
  supported yet.


.. _registering-types:

Registering types
=================

With the :func:`.register_type` function it is possible to register additional
types for use in jsonargparse parsers. If the type class can be instantiated
with a string representation and casting the instance to :code:`str` gives back
the string representation, then only the type class is given to
:func:`.register_type`. For example in the :code:`jsonargparse.typing` package
this is how complex numbers are registered: :code:`register_type(complex)`. For
other type classes that don't have these properties, to register it might be
necessary to provide a serializer and/or deserializer function. Including the
serializer and deserializer functions, the registration of the complex numbers
example is equivalent to :code:`register_type(complex, serializer=str,
deserializer=complex)`.

A more useful example could be registering the :code:`datetime` class. This case
requires to give both a serializer and a deserializer as seen below.

.. code-block:: python

    from datetime import datetime
    from jsonargparse import ArgumentParser
    from jsonargparse.typing import register_type

    def serializer(v):
        return v.isoformat()

    def deserializer(v):
        return datetime.strptime(v, '%Y-%m-%dT%H:%M:%S')

    register_type(datetime, serializer, deserializer)

    parser = ArgumentParser()
    parser.add_argument('--datetime', type=datetime)
    parser.parse_args(['--datetime=2008-09-03T20:56:35'])


.. _sub-classes:

Class type and sub-classes
==========================

It is also possible to use an arbitrary class as a type such that the argument
accepts this class or any derived subclass. In the config file a class is
represented by a dictionary with a :code:`class_path` entry indicating the dot
notation expression to import the class, and optionally some :code:`init_args`
that would be used to instantiate it. When parsing, it will be checked that the
class can be imported, that it is a subclass of the given type and that
:code:`init_args` values correspond to valid arguments to instantiate it. After
parsing, the config object will include the :code:`class_path` and
:code:`init_args` entries. To get a config object with all sub-classes
instantiated, the :py:meth:`.ArgumentParser.instantiate_subclasses` method is
used. The :code:`skip` parameter of the signature methods can also be used to
exclude arguments within subclasses. This is done by giving its relative
destination key, i.e. as :code:`param.init_args.subparam`.

A simple example would be having some config file :code:`config.yaml` as:

.. code-block:: yaml

    myclass:
      calendar:
        class_path: calendar.Calendar
        init_args:
          firstweekday: 1

Then in python:

.. code-block:: python

    >>> from calendar import Calendar
    >>> class MyClass:
            def __init__(self, calendar: Calendar):
                self.calendar = calendar
    >>> parser = ArgumentParser(parse_as_dict=True)
    >>> parser.add_class_arguments(MyClass, 'myclass')
    >>> cfg = parser.parse_path('config.yaml')
    >>> cfg['myclass']['calendar']
    {'class_path': 'calendar.Calendar', 'init_args': {'firstweekday': 1}}
    >>> cfg = parser.instantiate_subclasses(cfg)
    >>> cfg['myclass']['calendar'].getfirstweekday()
    1

In this example the :code:`class_path` points to the same class used for the
type. But a subclass of :code:`Calendar` with an extended list of init
parameters would also work. The help of the parser does not show details for a
type class since this depends on the subclass. To get help details for a
particular subclass there is a specific help option that receives the import
path. If there is some subclass of :code:`Calendar` which can be imported from
:code:`mycode.MyCalendar`, then it would be possible to see the corresponding
:code:`init_args` details by running the tool from the command line as:

.. code-block:: bash

    python tool.py --myclass.calendar.help mycode.MyCalendar

An individual argument can also be added having as type a class, i.e.
:code:`parser.add_argument('--calendar', type=Calendar)`. There is also another
method :py:meth:`.SignatureArguments.add_subclass_arguments` which does the same
as :code:`add_argument`, but has some added benefits: 1) the argument is added
in a new group automatically; 2) the argument values can be given in an
independent config file by specifying a path to it; and 3) by default sets a
useful :code:`metavar` and :code:`help` strings.

Default values
--------------

For a parameter that has a class as type it might also be wanted to set a
default value for it. Special care must be taken when doing this, could be
considered bad practice and be a good idea to avoid in most cases. The issue is
that classes are normally mutable. Depending on how the parameter value is used,
its default class instance in the signature could be changed. This goes against
what a default value is expected to be and lead to bugs which are difficult to
debug.

Since there are some legitimate use cases for class instances in defaults, they
are supported with a particular behavior and recommendations. The first approach
is using a normal class instance, for example:

.. code-block:: python

    class MyClass:
        def __init__(
            self,
            calendar: Calendar = Calendar(firstweekday=1),
        ):
            self.calendar = calendar

Adding this class to a parser will work without issues. Parsing would also work
and if not overridden the default class instance will be found in the respective
key of the config object. If :code:`--print_config` is used, the class instance
is just cast to a string. This means that the generated config file must be
modified to become a valid input to the parser. Due to the limitations and the
risk of mutable default this approach is discouraged.

The second approach which is the recommended one is to use the special function
:func:`.lazy_instance` to instantiate the default. Continuing with the same
example above this would be:

.. code-block:: python

    from jsonargparse import lazy_instance

    class MyClass:
        def __init__(
            self,
            calendar: Calendar = lazy_instance(Calendar, firstweekday=1),
        ):
            self.calendar = calendar

In this case the default value will still be an instance of :code:`Calendar`.
The difference is that the parsing methods would provide a dict with
:code:`class_path` and :code:`init_args` instead of the class instance.
Furthermore, if :py:meth:`.ArgumentParser.instantiate_subclasses` is used a new
instance of the class is created thereby avoiding issues related to the
mutability of the default.

Final classes
-------------

When a class is decorated with :func:`.final` there shouldn't be any derived
subclass. Using a final class as a type hint works similar to subclasses. The
difference is that the init args are given directly in a dictionary without
specifying a :code:`class_path`. Therefore, the code below would accept
the corresponding yaml structure.

.. code-block:: python

    from jsonargparse.typing import final

    @final
    class FinalCalendar(Calendar):
        pass

    parser = ArgumentParser()
    parser.add_argument('--calendar', type=FinalCalendar)
    cfg = parser.parse_path('config.yaml')

.. code-block:: yaml

    calendar:
      firstweekday: 1


.. _sub-commands:

Sub-commands
============

A way to define parsers in a modular way is what in argparse is known as
`sub-commands <https://docs.python.org/3/library/argparse.html#sub-commands>`__.
However, to promote modularity, in jsonargparse sub-commands work a bit
different than in argparse. To add sub-commands to a parser, the
:py:meth:`.ArgumentParser.add_subcommands` method is used. Then an existing
parser is added as a sub-command using :func:`.add_subcommand`. In a parsed
config object the sub-command will be stored in the :code:`subcommand` entry (or
whatever :code:`dest` was set to), and the values of the sub-command will be in
an entry with the same name as the respective sub-command. An example of
defining a parser with sub-commands is the following:

.. code-block:: python

    from jsonargparse import ArgumentParser
    ...
    parser_subcomm1 = ArgumentParser()
    parser_subcomm1.add_argument('--op1')
    ...
    parser_subcomm2 = ArgumentParser()
    parser_subcomm2.add_argument('--op2')
    ...
    parser = ArgumentParser(prog='app')
    parser.add_argument('--op0')
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand('subcomm1', parser_subcomm1)
    subcommands.add_subcommand('subcomm2', parser_subcomm2)

Then some examples of parsing are the following:

.. code-block:: python

    >>> parser.parse_args(['subcomm1', '--op1', 'val1'])
    Namespace(op0=None, subcomm1=Namespace(op1='val1'), subcommand='subcomm1')
    >>> parser.parse_args(['--op0', 'val0', 'subcomm2', '--op2', 'val2'])
    Namespace(op0='val0', subcomm2=Namespace(op2='val2'), subcommand='subcomm2')

Parsing config files with :py:meth:`.ArgumentParser.parse_path` or
:py:meth:`.ArgumentParser.parse_string` is also possible. Though there can only
be values for one of the sub-commands. The config file is not required to
specify a value for :code:`subcommand`. For the example parser above a valid
yaml would be:

.. code-block:: yaml

    # File: example.yaml
    op0: val0
    subcomm1:
      op1: val1

Parsing of environment variables works similar to :class:`.ActionParser`. For
the example parser above, all environment variables for :code:`subcomm1` would
have as prefix :code:`APP_SUBCOMM1_` and likewise for :code:`subcomm2` as prefix
:code:`APP_SUBCOMM2_`. The sub-command to use could be chosen by setting
environment variable :code:`APP_SUBCOMMAND`.

It is possible to have multiple levels of sub-commands. With multiple levels
there is one basic requirement: the sub-commands must be added in the order of
the levels. This is, first call :func:`add_subcommands` and
:func:`add_subcommand` for the first level. Only after do the same for the
second level, and so on.


.. _json-schemas:

Json schemas
============

The :class:`.ActionJsonSchema` class is provided to allow parsing and validation
of values using a json schema. This class requires the `jsonschema
<https://pypi.org/project/jsonschema/>`__ python package. Though note that
jsonschema is not a requirement of the minimal jsonargparse install. To enable
this functionality install with the :code:`jsonschema` extras require as
explained in section :ref:`installation`.

Check out the `jsonschema documentation
<https://python-jsonschema.readthedocs.io/>`__ to learn how to write a schema.
The current version of jsonargparse uses Draft7Validator. Parsing an argument
using a json schema is done like in the following example:

.. code-block:: python

    >>> schema = {
    ...     "type" : "object",
    ...     "properties" : {
    ...         "price" : {"type" : "number"},
    ...         "name" : {"type" : "string"},
    ...     },
    ... }

    >>> from jsonargparse import ActionJsonSchema
    >>> parser.add_argument('--op', action=ActionJsonSchema(schema=schema))

    >>> parser.parse_args(['--op', '{"price": 1.5, "name": "cookie"}'])
    Namespace(op=Namespace(name='cookie', price=1.5))

Instead of giving a json string as argument value, it is also possible to
provide a path to a json/yaml file, which would be loaded and validated against
the schema. If the schema defines default values, these will be used by the
parser to initialize the config values that are not specified. When adding an
argument with the :class:`.ActionJsonSchema` action, you can use "%s" in the
:code:`help` string so that in that position the schema is printed.


.. _jsonnet-files:

Jsonnet files
=============

The Jsonnet support requires `jsonschema
<https://pypi.org/project/jsonschema/>`__ and `jsonnet
<https://pypi.org/project/jsonnet/>`__ python packages which are not included
with minimal jsonargparse install. To enable this functionality install
jsonargparse with the :code:`jsonnet` extras require as explained in section
:ref:`installation`.

By default an :class:`.ArgumentParser` parses configuration files as yaml.
However, if instantiated giving as argument :code:`parser_mode='jsonnet'`, then
:func:`parse_args`, :func:`parse_path` and :func:`parse_string` will expect
config files to be in jsonnet format instead. Example:

.. code-block:: python

    >>> from jsonargparse import ArgumentParser, ActionConfigFile
    >>> parser = ArgumentParser(parser_mode='jsonnet')
    >>> parser.add_argument('--cfg', action=ActionConfigFile)
    >>> cfg = parser.parse_args(['--cfg', 'example.jsonnet'])

Jsonnet files are commonly parametrized, thus requiring external variables for
parsing. For these cases, instead of changing the parser mode away from yaml,
the :class:`.ActionJsonnet` class can be used. This action allows to define an
argument which would be a jsonnet string or a path to a jsonnet file. Moreover,
another argument can be specified as the source for any external variables
required, which would be either a path to or a string containing a json
dictionary of variables. Its use would be as follows:

.. code-block:: python

    from jsonargparse import ArgumentParser, ActionJsonnet, ActionJsonnetExtVars
    parser = ArgumentParser()
    parser.add_argument('--in_ext_vars',
        action=ActionJsonnetExtVars())
    parser.add_argument('--in_jsonnet',
        action=ActionJsonnet(ext_vars='in_ext_vars'))

For example, if a jsonnet file required some external variable :code:`param`,
then the jsonnet and the external variable could be given as:

.. code-block:: python

        cfg = parser.parse_args(['--in_ext_vars', '{"param": 123}',
                                 '--in_jsonnet', 'path_to_jsonnet'])

Note that the external variables argument must be provided before the jsonnet
path so that this dictionary already exists when parsing the jsonnet.

The :class:`.ActionJsonnet` class also accepts as argument a json schema, in
which case the jsonnet would be validated against this schema right after
parsing.


.. _parsing-paths:

Parsing paths
=============

For some use cases it is necessary to parse file paths, checking its existence
and access permissions, but not necessarily opening the file. Moreover, a file
path could be included in a config file as relative with respect to the config
file's location. After parsing it should be easy to access the parsed file path
without having to consider the location of the config file. To help in these
situations jsonargparse includes a type generator :func:`.path_type`, some
predefined types (e.g. :class:`.Path_fr`) and the :class:`.ActionPathList`
class.

For example suppose you have a directory with a configuration file
:code:`app/config.yaml` and some data :code:`app/data/info.db`. The contents of
the yaml file is the following:

.. code-block:: yaml

    # File: config.yaml
    databases:
      info: data/info.db

To create a parser that checks that the value of :code:`databases.info` is a
file that exists and is readable, the following could be done:

.. code-block:: python

    from jsonargparse import ArgumentParser
    from jsonargparse.typing import Path_fr
    parser = ArgumentParser()
    parser.add_argument('--databases.info', type=Path_fr)
    cfg = parser.parse_path('app/config.yaml')

The :code:`fr` in the type are flags that stand for file and readable. After
parsing, the value of :code:`databases.info` will be an instance of the
:class:`.Path` class that allows to get both the original relative path as
included in the yaml file, or the corresponding absolute path:

.. code-block:: python

    >>> str(cfg.databases.info)
    'data/info.db'
    >>> cfg.databases.info()
    '/YOUR_CWD/app/data/info.db'

Likewise directories can be parsed using the :class:`.Path_dw` type, which would
require a directory to exist and be writeable. New path types can be created
using the :func:`.path_type` function. For example to create a type for files
that must exist and be both readable and writeable, the command would be
:code:`Path_frw = path_type('frw')`. If the file :code:`app/config.yaml` is not
writeable, then using the type to cast :code:`Path_frw('app/config.yaml')` would
raise a *TypeError: File is not writeable* exception. For more information of
all the mode flags supported, refer to the documentation of the :class:`.Path`
class.

The content of a file that a :class:`.Path` instance references can be read by
using the :py:meth:`.Path.get_content` method. For the previous example would be
:code:`info_db = cfg.databases.info.get_content()`.

An argument with a path type can be given :code:`nargs='+'` to parse multiple
paths. But it might also be wanted to parse a list of paths found in a plain
text file or from stdin. For this the :class:`.ActionPathList` is used and as
argument either the path to a file listing the paths is given or the special
:code:`'-'` string for reading the list from stdin. Example:

.. code-block:: python

    from jsonargparse import ActionPathList
    parser.add_argument('--list', action=ActionPathList(mode='fr'))
    cfg = parser.parse_args(['--list', 'paths.lst')  # Text file with paths
    cfg = parser.parse_args(['--list', '-')          # List from stdin

If :code:`nargs='+'` is given to :code:`add_argument` with
:class:`.ActionPathList` then a single list is generated including all paths in
all provided lists.

Note: the :class:`.Path` class is currently not fully supported in windows.


.. _parsing-urls:

Parsing URLs
============

The :func:`.path_type` function also supports URLs which after parsing the
:py:meth:`.Path.get_content` method can be used to perform a GET request to the
corresponding URL and retrieve its content. For this to work the *validators*
and *requests* python packages are required. Alternatively, :func:`.path_type`
can also be used for `fsspec <https://filesystem-spec.readthedocs.io>`__
supported file systems. The respective optional package(s) will be installed
along with jsonargparse if installed with the :code:`urls` or :code:`fsspec`
extras require as explained in section :ref:`installation`.

The :code:`'u'` flag is used to parse URLs using requests and the flag
:code:`'s'` to parse fsspec file systems. For example if it is desired that an
argument can be either a readable file or URL, the type would be created as
:code:`Path_fur = path_type('fur')`. If the value appears to be a URL according
to :func:`validators.url.url` then a HEAD request would be triggered to check if
it is accessible. To get the content of the parsed path, without needing to care
if it is a local file or a URL, the :py:meth:`.Path.get_content` can be used.

If you import :code:`from jsonargparse import set_config_read_mode` and then run
:code:`set_config_read_mode(urls_enabled=True)` or
:code:`set_config_read_mode(fsspec_enabled=True)`, the following functions and
classes will also support loading from URLs:
:py:meth:`.ArgumentParser.parse_path`, :py:meth:`.ArgumentParser.get_defaults`
(:code:`default_config_files` argument), :class:`.ActionConfigFile`,
:class:`.ActionJsonSchema`, :class:`.ActionJsonnet` and :class:`.ActionParser`.
This means that a tool that can receive a configuration file via
:class:`.ActionConfigFile` is able to get the content from a URL, thus something
like the following would work:

.. code-block:: bash

    $ my_tool.py --cfg http://example.com/config.yaml


.. _restricted-numbers:

Restricted numbers
==================

It is quite common that when parsing a number, its range should be limited. To
ease these cases the module :code:`jsonargparse.typing` includes some predefined
types and a function :func:`.restricted_number_type` to define new types. The
predefined types are: :class:`.PositiveInt`, :class:`.NonNegativeInt`,
:class:`.PositiveFloat`, :class:`.NonNegativeFloat`,
:class:`.ClosedUnitInterval` and :class:`.OpenUnitInterval`. Examples of usage
are:

.. code-block:: python

    from jsonargparse.typing import PositiveInt, PositiveFloat, restricted_number_type
    # float larger than zero
    parser.add_argument('--op1', type=PositiveFloat)
    # between 0 and 10
    from_0_to_10 = restricted_number_type('from_0_to_10', int, [('>=', 0), ('<=', 10)])
    parser.add_argument('--op2', type=from_0_to_10)
    # either int larger than zero or 'off' string
    def int_or_off(x): return x if x == 'off' else PositiveInt(x)
    parser.add_argument('--op3', type=int_or_off))


.. _restricted-strings:

Restricted strings
==================

Similar to the restricted numbers, there is a function to create string types
that are restricted to match a given regular expression:
:func:`.restricted_string_type`. A predefined type is :class:`.Email` which is
restricted so that it follows the normal email pattern. For example to add an
argument required to be exactly four uppercase letters:

.. code-block:: python

    from jsonargparse.typing import Email, restricted_string_type
    CodeType = restricted_string_type('CodeType', '^[A-Z]{4}$')
    parser.add_argument('--code', type=CodeType)
    parser.add_argument('--email', type=Email)


Enum arguments
==============

Another case of restricted values is string choices. In addition to the common
:code:`choices` given as a list of strings, it is also possible to provide as
type an :code:`Enum` class. This has the added benefit that strings are mapped
to some desired values. For example:

.. code-block:: python

    >>> class MyEnum(enum.Enum):
    ...     choice1 = -1
    ...     choice2 = 0
    ...     choice3 = 1
    >>> parser.add_argument('--op', type=MyEnum)
    >>> parser.parse_args(['--op=choice1'])
    Namespace(op=<MyEnum.choice1: -1>)


.. _boolean-arguments:

Boolean arguments
=================

Parsing boolean arguments is very common, however, the original argparse only
has a limited support for them, via :code:`store_true` and :code:`store_false`.
Futhermore unexperienced users might mistakenly use :code:`type=bool` which
would not provide the intended behavior.

With jsonargparse adding an argument with :code:`type=bool` the intended action
is implemented. If given as values :code:`{'yes', 'true'}` or :code:`{'no',
'false'}` the corresponding parsed values would be :code:`True` or
:code:`False`. For example:

.. code-block:: python

    >>> parser.add_argument('--op1', type=bool, default=False)
    >>> parser.add_argument('--op2', type=bool, default=True)
    >>> parser.parse_args(['--op1', 'yes', '--op2', 'false'])
    Namespace(op1=True, op2=False)

Sometimes it is also useful to define two paired options, one to set
:code:`True` and the other to set :code:`False`. The :class:`.ActionYesNo` class
makes this straightforward. A couple of examples would be:

.. code-block:: python

    from jsonargparse import ActionYesNo
    # --opt1 for true and --no_opt1 for false.
    parser.add_argument('--op1', action=ActionYesNo)
    # --with-opt2 for true and --without-opt2 for false.
    parser.add_argument('--with-op2', action=ActionYesNo(yes_prefix='with-', no_prefix='without-'))

If the :class:`.ActionYesNo` class is used in conjunction with :code:`nargs='?'`
the options can also be set by giving as value any of :code:`{'true', 'yes',
'false', 'no'}`.


.. _parser-arguments:

Parsers as arguments
====================

Sometimes it is useful to take an already existing parser that is required
standalone in some part of the code, and reuse it to parse an inner node of
another more complex parser. For these cases an argument can be defined using
the :class:`.ActionParser` class. An example of how to use this class is the
following:

.. code-block:: python

    from jsonargparse import ArgumentParser, ActionParser
    inner_parser = ArgumentParser(prog='app1')
    inner_parser.add_argument('--op1')
    ...
    outer_parser = ArgumentParser(prog='app2')
    outer_parser.add_argument('--inner.node',
        title='Inner node title',
        action=ActionParser(parser=inner_parser))

When using the :class:`.ActionParser` class, the value of the node in a config
file can be either the complex node itself, or the path to a file which will be
loaded and parsed with the corresponding inner parser. Naturally using
:class:`.ActionConfigFile` to parse a complete config file will parse the inner
nodes correctly.

Note that when adding :code:`inner_parser` a title was given. In the help, the
added parsers are shown as independent groups starting with the given
:code:`title`. It is also possible to provide a :code:`description`.

Regarding environment variables, the prefix of the outer parser will be used to
populate the leaf nodes of the inner parser. In the example above, if
:code:`inner_parser` is used to parse environment variables, then as normal
:code:`APP1_OP1` would be checked to populate option :code:`op1`. But if
:code:`outer_parser` is used, then :code:`APP2_INNER__NODE__OP1` would be
checked to populate :code:`inner.node.op1`.

An important detail to note is that the parsers that are given to
:class:`.ActionParser` are internally modified. Therefore, to use the parser
both as standalone and an inner node, it is necessary to implement a function
that instantiates the parser. This function would be used in one place to get an
instance of the parser for standalone parsing, and in some other place use the
function to provide an instance of the parser to :class:`.ActionParser`.


.. _tab-completion:

Tab completion
==============

Tab completion is available for jsonargparse parsers by using the `argcomplete
<https://pypi.org/project/argcomplete/>`__ package. There is no need to
implement completer functions or to call :func:`argcomplete.autocomplete` since
this is done automatically by :py:meth:`.ArgumentParser.parse_args`. The only
requirement to enable tab completion is to install argcomplete either directly
or by installing jsonargparse with the :code:`argcomplete` extras require as
explained in section :ref:`installation`. Then the tab completion can be enabled
`globally <https://kislyuk.github.io/argcomplete/#global-completion>`__ for all
argcomplete compatible tools or for each `individual
<https://kislyuk.github.io/argcomplete/#synopsis>`__ tool. A simple
:code:`example.py` tool would be:

.. code-block:: python

    #!/usr/bin/env python3

    from typing import Optional
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--bool', type=Optional[bool])

    parser.parse_args()

Then in a bash shell you can add the executable bit to the script, activate tab
completion and use it as follows:

.. code-block:: bash

    $ chmod +x example.py
    $ eval "$(register-python-argcomplete example.py)"

    $ ./example.py --bool <TAB><TAB>
    false  null   true
    $ ./example.py --bool f<TAB>
    $ ./example.py --bool false


.. _logging:

Logging
=======

The parsers from jsonargparse log some basic events, though by default this is
disabled. To enable it the :code:`logger` argument should be set when creating
an :class:`.ArgumentParser` object. The intended use is to give as value an
already existing logger object which is used for the whole application. Though
for convenience to enable a default logger the :code:`logger` argument can also
receive :code:`True` or a string which sets the name of the logger or a
dictionary that can include the name and the level, e.g. :code:`{"name":
"myapp", "level": "ERROR"}`. If `reconplogger
<https://pypi.org/project/reconplogger/>`__ is installed, setting :code:`logger`
to :code:`True` or a dictionary without specifying a name, then the reconplogger
is used.


Contributing
============

Contributions to jsonargparse are very welcome, be it just to create `issues
<https://github.com/omni-us/jsonargparse/issues>`_ for reporting bugs and
proposing enhancements, or more directly by creating `pull requests
<https://github.com/omni-us/jsonargparse/pulls>`_.

If you intend to work with the source code, note that this project does not
include any :code:`requirements.txt` file. This is by intention. To make it very
clear what are the requirements for different use cases, all the requirements of
the project are stored in the file :code:`setup.cfg`. The basic runtime
requirements are defined in section :code:`[options]` in the
:code:`install_requires` entry. All extras requires for optional features listed
in :ref:`installation` are stored in section :code:`[options.extras_require]`.
Also there are :code:`test`, :code:`test_no_urls`, :code:`dev` and :code:`doc`
entries in the same :code:`[options.extras_require]` section which lists
requirements for testing, development and documentation building.

The recommended way to work with the source code is the following. First clone
the repository, then create a virtual environment, activate it and finally
install the development requirements. More precisely the steps are:

.. code-block:: bash

    git clone https://github.com/omni-us/jsonargparse.git
    cd jsonargparse
    virtualenv -p python3 venv
    . venv/bin/activate

The crucial step is installing the requirements which would be done by running:

.. code-block:: bash

    pip install -e ".[dev,all]"

Running the unit tests can be done either using using `tox
<https://tox.readthedocs.io/en/stable/>`__ or the :code:`setup.py` script. The
unit tests are also installed with the package, thus can be used to in a
production system.

.. code-block:: bash

    tox  # Run tests using tox
    ./setup.py test_coverage  # Run tests and generate coverage report
    python3 -m jsonargparse_tests  # Run tests for installed package
