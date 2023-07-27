.. image:: https://readthedocs.org/projects/jsonargparse/badge/?version=stable
    :target: https://readthedocs.org/projects/jsonargparse/
.. image:: https://github.com/omni-us/jsonargparse/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/omni-us/jsonargparse/actions/workflows/tests.yml
.. image:: https://codecov.io/gh/omni-us/jsonargparse/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/omni-us/jsonargparse
.. image:: https://sonarcloud.io/api/project_badges/measure?project=omni-us_jsonargparse&metric=alert_status
    :target: https://sonarcloud.io/dashboard?id=omni-us_jsonargparse
.. image:: https://badge.fury.io/py/jsonargparse.svg
    :target: https://badge.fury.io/py/jsonargparse


jsonargparse
============

Docs: https://jsonargparse.readthedocs.io/

Source: https://github.com/omni-us/jsonargparse/

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

- Great support of type hints for automatic creation of parsers and minimal
  boilerplate command line interfaces.

- Non-intrusive/decoupled parsing logic. No need to inherit from a special class
  or add decorators or use custom type hints.

- Easy to implement configurable dependency injection (object composition).

- Support for nested namespaces making possible to parse config files with
  non-flat hierarchies.

- Parsing of relative paths within config files and path lists.

- Support for two popular supersets of json: yaml and jsonnet.

- Parsers can be configured just like with python's argparse, thus it has a
  gentle learning curve.

- Several convenient types and action classes to ease common parsing use cases
  (paths, comparison operators, json schemas, enums, regex strings, ...).

- Support for command line tab argument completion using `argcomplete
  <https://pypi.org/project/argcomplete/>`__.


.. _installation:

Installation
============

You can install using `pip <https://pypi.org/project/jsonargparse/>`__ as:

.. code-block:: bash

    pip install jsonargparse

By default the only dependency that jsonargparse installs is `PyYAML
<https://pypi.org/project/PyYAML/>`__. However, several optional features can be
enabled by specifying any of the following extras requires: ``signatures``,
``jsonschema``, ``jsonnet``, ``urls``, ``argcomplete`` and ``reconplogger``.
There is also the ``all`` extras require to enable all optional features.
Installing jsonargparse with extras require is as follows:

.. code-block:: bash

    pip install "jsonargparse[signatures,urls]"  # Enable signatures and URLs features
    pip install "jsonargparse[all]"              # Enable all optional features

The following table references sections that describe optional features and the
corresponding extras requires that enables them.

+----------------------------------+-------------+-------------+---------+------------+------------+
|                                  | urls/fsspec | argcomplete | jsonnet | jsonschema | signatures |
+----------------------------------+-------------+-------------+---------+------------+------------+
| :ref:`classes-methods-functions` |             |             |         |            | ✓          |
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

There are multiple ways of using jsonargparse. One is to construct low level
parsers (see :ref:`parsers`) being almost a drop in replacement of argparse.
However, argparse is too verbose and leads to unnecessary duplication. The
simplest and recommended way of using jsonargparse is by using the :func:`.CLI`
function, which has the benefit of minimizing boilerplate code. A simple example
is:

.. testcode::

    from jsonargparse import CLI


    def command(name: str, prize: int = 100):
        """Prints the prize won by a person.

        Args:
            name: Name of winner.
            prize: Amount won.
        """
        print(f"{name} won {prize}€!")


    if __name__ == "__main__":
        CLI(command)

Note that the ``name`` and ``prize`` parameters have type hints and are
described in the docstring. These are shown in the help of the command line
tool. In a shell you could see the help and run a command as follows:

.. code-block:: bash

    $ python example.py --help
    ...
    Prints the prize won by a person:
      name                  Name of winner. (required, type: str)
      --prize PRIZE         Amount won. (type: int, default: 100)

    $ python example.py Lucky --prize=1000
    Lucky won 1000€!

.. note::

    Parsing of docstrings is an optional feature. For this example to work as
    shown, jsonargparse needs to be installed with the ``signatures`` extras
    require as explained in section :ref:`installation`.

When :func:`.CLI` receives a single class, the first arguments are for
parameters to instantiate the class, then a method name is expected (i.e.
methods become :ref:`sub-commands`) and the remaining arguments are for
parameters of this method. An example would be:

.. testcode::

    from random import randint
    from jsonargparse import CLI


    class Main:
        def __init__(self, max_prize: int = 100):
            """
            Args:
                max_prize: Maximum prize that can be awarded.
            """
            self.max_prize = max_prize

        def person(self, name: str):
            """
            Args:
                name: Name of winner.
            """
            return f"{name} won {randint(0, self.max_prize)}€!"


    if __name__ == "__main__":
        print(CLI(Main))

Then in a shell you could run:

.. code-block:: bash

    $ python example.py --max_prize=1000 person Lucky
    Lucky won 632€!

.. doctest:: :hide:

    >>> CLI(Main, args=["--max_prize=1000", "person", "Lucky"])  # doctest: +ELLIPSIS
    'Lucky won ...€!'

If the class given does not have any methods, there will be no sub-commands and
:func:`.CLI` will return an instance of the class. For example:

.. testcode::

    from dataclasses import dataclass
    from jsonargparse import CLI


    @dataclass
    class Settings:
        name: str
        prize: int = 100


    if __name__ == "__main__":
        print(CLI(Settings, as_positional=False))

Then in a shell you could run:

.. code-block:: bash

    $ python example.py --name=Lucky
    Settings(name='Lucky', prize=100)

.. doctest:: :hide:

    >>> CLI(Settings, as_positional=False, args=["--name=Lucky"])  # doctest: +ELLIPSIS
    Settings(name='Lucky', prize=100)

Note the use of ``as_positional=False`` to make required arguments as
non-positional.

If more than one function is given to :func:`.CLI`, then any of them can be run
via :ref:`sub-commands` similar to the single class example above, i.e.
``example.py function [arguments]`` where ``function`` is the name of the
function to execute. If multiple classes or a mixture of functions and classes
is given to :func:`.CLI`, to execute a method of a class, two levels of
:ref:`sub-commands` are required. The first sub-command would be the name of the
class and the second the name of the method, i.e. ``example.py class
[init_arguments] method [arguments]``.

.. note::

    The examples above are extremely simple, only defining parameters with
    ``str`` and ``int`` type hints. The true power of jsonargparse is its
    support for a wide range of types, see :ref:`type-hints`. It is even
    possible to use general classes as type hints, allowing to easily implement
    configurable `dependency injection (object composition)
    <https://en.wikipedia.org/wiki/Dependency_injection>`__, see
    :ref:`sub-classes`.

Writing configuration files
---------------------------

All tools implemented with the :func:`.CLI` function have the ``--config``
option to provide settings in a config file (more details in
:ref:`configuration-files`). This becomes very useful when the number of
configurable parameters is large. To ease the writing of config files, there is
also the option ``--print_config`` which prints to standard output a yaml with
all settings that the tool supports with their default values. Users of the tool
can be advised to follow the following steps:

.. code-block:: bash

    # Dump default configuration to have as reference
    python example.py --print_config > config.yaml
    # Modify the config as needed (all default settings can be removed)
    nano config.yaml
    # Run the tool using the adapted config
    python example.py --config config.yaml

Comparison to Fire
------------------

The :func:`.CLI` feature is similar to and inspired by `Fire
<https://pypi.org/project/fire/>`__. However, there are fundamental differences.
First, the purpose is not allowing to call any python object from the command
line. It is only intended for running functions and classes specifically written
for this purpose. Second, the arguments are expected to have type hints, and the
given values will be validated according to these. Third, the return values of
the functions are not automatically printed. :func:`.CLI` returns the value and
it is up to the developer to decide what to do with it.


.. _tutorials:

Tutorials
=========

- `"jsonargparse - Say goodbye to configuration hassles"
  <https://2022.pycon.de/program/XK73C3/>`__  by Marianne Stecklina at PyCon DE
  & PyData Berlin 2022

    - Presentation video: https://youtu.be/2gDf2S0nHKg
    - GitHub repository: https://github.com/stecklin/pycon22-jsonargparse


.. _parsers:

Parsers
=======

An argument parser is created just like it is done with python's `argparse
<https://docs.python.org/3/library/argparse.html>`__. You import the module,
create a parser object and then add arguments to it. A simple example would be:

.. testcode::

    from jsonargparse import ArgumentParser

    parser = ArgumentParser(prog="app", description="Description for my app.")
    parser.add_argument("--opt1", type=int, default=0, help="Help for option 1.")
    parser.add_argument("--opt2", type=float, default=1.0, help="Help for option 2.")


After creating the parser, you can use it to parse command line arguments with
the :py:meth:`.ArgumentParser.parse_args` function, after which you get
an object with the parsed values or defaults available as attributes. For
illustrative purposes giving to :func:`parse_args` a list of arguments (instead
of automatically getting them from the command line arguments), with the parser
shown above you would observe:

.. doctest::

    >>> cfg = parser.parse_args(["--opt2", "2.3"])
    >>> cfg.opt1, type(cfg.opt1)
    (0, <class 'int'>)
    >>> cfg.opt2, type(cfg.opt2)
    (2.3, <class 'float'>)

If the parsing fails the standard behavior is that the usage is printed and the
program is terminated. Alternatively you can initialize the parser with
``exit_on_error=False`` in which case an :class:`.ArgumentError` is raised.


Override order
--------------

Final parsed values depend on different sources, namely: source code, command
line arguments, :ref:`configuration-files` and :ref:`environment-variables`.
Values are overridden based on the following precedence:

1. Defaults defined in the source code.
2. Existing default config files in the order defined in
   ``default_config_files``, e.g. ``~/.config/myapp.yaml``.
3. Full config environment variable, e.g. ``APP_CONFIG``.
4. Individual key environment variables, e.g. ``APP_OPT1``.
5. Command line arguments in order left to right (might include config files).

Depending on the parse method used (see :class:`.ArgumentParser`) and how the
parser was built, some of the options above might not apply. Parsing of
environment variables must be explicitly enabled, except if using
:py:meth:`.ArgumentParser.parse_env`. If the parser does not have an
:class:`.ActionConfigFile` argument, then there is no parsing of a full config
environment variable or a way to provide a config file from command line.


Capturing parsers
-----------------

It can be common practice to have a function that implements an entire CLI or a
function that constructs a parser conditionally based on some parameters and
then parses. For example, one might have:

.. testcode::

    from jsonargparse import ArgumentParser


    def main_cli():
        parser = ArgumentParser()
        ...
        cfg = parser.parse_args()
        ...


    if __name__ == "__main__":
        main_cli()

For some use cases it is necessary to get an instance of the parser object,
without doing any parsing. For instance `sphinx-argparse
<https://sphinx-argparse.readthedocs.io/en/stable/>`__ can be used to include
the help of CLIs in automatically generated documentation of a package. To use
sphinx-argparse it is necessary to have a function that returns the parser.
Having a CLI function this could be easily implemented with
:func:`.capture_parser` as follows:

.. testcode::

    from jsonargparse import capture_parser


    def get_parser():
        return capture_parser(main_cli)

.. note::

    The official way to obtain the parser for command line tools based on
    :func:`.CLI` is by using :func:`.capture_parser`.


.. _type-hints:

Type hints
==========

An important feature of jsonargparse is a wide support the argument types and
respective validation. This extended support makes use of Python's type hint
syntax. For example, an argument that can be ``None`` or a float in the range
``(0, 1)`` or a positive int could be added using a type hint as follows:

.. testcode::

    from typing import Optional, Union
    from jsonargparse.typing import PositiveInt, OpenUnitInterval

    parser.add_argument("--op", type=Optional[Union[PositiveInt, OpenUnitInterval]])

The types in :py:mod:`jsonargparse.typing` are included for convenience since
they are useful in argument parsing use cases and not available in standard
python. However, there is no need to use jsonargparse specific types.

A wide range of type hints are supported and with arbitrary complexity/nesting.
Some notes about this support are:

- Nested types are supported as long as at least one child type is supported. By
  nesting it is meant child types inside ``List``, ``Dict``, etc. There is no
  limit in nesting depth.

- Postponed evaluation of types PEP `563 <https://peps.python.org/pep-0563/>`__
  (i.e. ``from __future__ import annotations``) is supported. Also supported on
  ``python<=3.9`` are PEP `585 <https://peps.python.org/pep-0585/>`__ (i.e.
  ``list[<type>], dict[<type>], ...`` instead of ``List[<type>], Dict[<type>],
  ...``) and `604 <https://peps.python.org/pep-0604/>`__ (i.e. ``<type> |
  <type>`` instead of ``Union[<type>, <type>]``).

- Fully supported types are: ``str``, ``bool`` (more details in
  :ref:`boolean-arguments`), ``int``, ``float``, ``complex``,
  ``bytes``/``bytearray`` (Base64 encoding), ``range``, ``List`` (more details
  in :ref:`list-append`), ``Iterable``, ``Sequence``, ``Any``, ``Union``,
  ``Optional``, ``Type``, ``Enum``, ``PathLike``, ``UUID``, ``timedelta``,
  restricted types as explained in sections :ref:`restricted-numbers` and
  :ref:`restricted-strings` and paths and URLs as explained in sections
  :ref:`parsing-paths` and :ref:`parsing-urls`.

- ``Dict``, ``Mapping``, and ``MutableMapping`` are supported but only with
  ``str`` or ``int`` keys. For more details see :ref:`dict-items`.

- ``Tuple``, ``Set`` and ``MutableSet`` are supported even though they can't be
  represented in json distinguishable from a list. Each ``Tuple`` element
  position can have its own type and will be validated as such. ``Tuple`` with
  ellipsis (``Tuple[type, ...]``) is also supported. In command line arguments,
  config files and environment variables, tuples and sets are represented as an
  array.

- To set a value to ``None`` it is required to use ``null`` since this is how
  json/yaml defines it. To avoid confusion in the help, ``NoneType`` is
  displayed as ``null``. For example a function argument with type and default
  ``Optional[str] = None`` would be shown in the help as ``type: Union[str,
  null], default: null``.

- Normal classes can be used as a type, which are specified with a dict
  containing ``class_path`` and optionally ``init_args``.
  :py:meth:`.ArgumentParser.instantiate_classes` can be used to instantiate all
  classes in a config object. For more details see :ref:`sub-classes`.

- ``dataclasses`` are supported even when nested. Final classes, attrs'
  ``define`` decorator, and pydantic's ``dataclass`` decorator and ``BaseModel``
  classes are supported and behave like standard dataclasses. For more details
  see :ref:`dataclass-like`. If a dataclass is mixed inheriting from a normal
  class, it is considered a subclass type instead of a dataclass.

- `Pydantic types <https://docs.pydantic.dev/usage/types/#pydantic-types>`__ are
  supported. There might be edge cases which don't work as expected. Please
  report any encountered issues.

- ``Callable`` is supported by either giving a dot import path to a callable
  object or by giving a dict with a ``class_path`` and optionally ``init_args``
  entries. The specified class must either instantiate into a callable or be a
  subclass of the return type of the callable. For these cases running
  :py:meth:`.ArgumentParser.instantiate_classes` will instantiate the class or
  provide a function that returns the instance of the class. For more details
  see :ref:`callable-type`. Currently the callable's argument and return types
  are not validated.


.. _restricted-numbers:

Restricted numbers
------------------

It is quite common that when parsing a number, its range should be limited. To
ease these cases the module ``jsonargparse.typing`` includes some predefined
types and a function :func:`.restricted_number_type` to define new types. The
predefined types are: :class:`.PositiveInt`, :class:`.NonNegativeInt`,
:class:`.PositiveFloat`, :class:`.NonNegativeFloat`,
:class:`.ClosedUnitInterval` and :class:`.OpenUnitInterval`. Examples of usage
are:

.. testcode::

    from jsonargparse.typing import PositiveInt, PositiveFloat, restricted_number_type

    # float larger than zero
    parser.add_argument("--op1", type=PositiveFloat)
    # between 0 and 10
    from_0_to_10 = restricted_number_type("from_0_to_10", int, [(">=", 0), ("<=", 10)])
    parser.add_argument("--op2", type=from_0_to_10)


    # either int larger than zero or 'off' string
    def int_or_off(x):
        return x if x == "off" else PositiveInt(x)


    parser.add_argument("--op3", type=int_or_off)


.. _restricted-strings:

Restricted strings
------------------

Similar to the restricted numbers, there is a function to create string types
that are restricted to match a given regular expression:
:func:`.restricted_string_type`. A predefined type is :class:`.Email` which is
restricted so that it follows the normal email pattern. For example to add an
argument required to be exactly four uppercase letters:

.. testcode::

    from jsonargparse.typing import Email, restricted_string_type

    CodeType = restricted_string_type("CodeType", "^[A-Z]{4}$")
    parser.add_argument("--code", type=CodeType)
    parser.add_argument("--email", type=Email)


.. _parsing-paths:

Parsing paths
-------------

For some use cases it is necessary to parse file paths, checking its existence
and access permissions, but not necessarily opening the file. Moreover, a file
path could be included in a config file as relative with respect to the config
file's location. After parsing it should be easy to access the parsed file path
without having to consider the location of the config file. To help in these
situations jsonargparse includes a type generator :func:`.path_type`, some
predefined types (e.g. :class:`.Path_fr`).

For example suppose you have a directory with a configuration file
``app/config.yaml`` and some data ``app/data/info.db``. The contents of the yaml
file is the following:

.. code-block:: yaml

    # File: config.yaml
    databases:
      info: data/info.db

To create a parser that checks that the value of ``databases.info`` is a file
that exists and is readable, the following could be done:

.. testsetup:: paths

    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="_jsonargparse_doctest_")
    os.chdir(tmpdir)
    os.mkdir("app")
    os.mkdir("app/data")
    with open("app/config.yaml", "w") as f:
        f.write("databases:\n  info: data/info.db\n")
    with open("app/data/info.db", "w") as f:
        f.write("info\n")

.. testcleanup:: paths

    os.chdir(cwd)
    shutil.rmtree(tmpdir)

.. testcode:: paths

    from jsonargparse import ArgumentParser
    from jsonargparse.typing import Path_fr

    parser = ArgumentParser()
    parser.add_argument("--databases.info", type=Path_fr)
    cfg = parser.parse_path("app/config.yaml")

The ``fr`` in the type are flags that stand for file and readable. After
parsing, the value of ``databases.info`` will be an instance of the
:class:`.Path` class that allows to get both the original relative path as
included in the yaml file, or the corresponding absolute path:

.. doctest:: paths

    >>> str(cfg.databases.info)
    'data/info.db'
    >>> cfg.databases.info()  # doctest: +ELLIPSIS
    '/.../app/data/info.db'

Likewise directories can be parsed using the :class:`.Path_dw` type, which would
require a directory to exist and be writeable. New path types can be created
using the :func:`.path_type` function. For example to create a type for files
that must exist and be both readable and writeable, the command would be
``Path_frw = path_type('frw')``. If the file ``app/config.yaml`` is not
writeable, then using the type to cast ``Path_frw('app/config.yaml')`` would
raise a *TypeError: File is not writeable* exception. For more information of
all the mode flags supported, refer to the documentation of the :class:`.Path`
class.

The content of a file that a :class:`.Path` instance references can be read by
using the :py:meth:`.Path.get_content` method. For the previous example would be
``info_db = cfg.databases.info.get_content()``.

An argument with a path type can be given ``nargs='+'`` to parse multiple paths.
But it might also be wanted to parse a list of paths found in a plain text file
or from stdin. For this add the argument with type ``List[<path_type>]`` and
``enable_path=True``. To read from stdin give the special string ``'-'``.
Example:

.. testsetup:: path_list

    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="_jsonargparse_doctest_")
    os.chdir(tmpdir)
    pathlib.Path("paths.lst").write_text("paths.lst\n")

    parser = ArgumentParser()

    stdin = sys.stdin
    sys.stdin = StringIO("paths.lst\n")

.. testcleanup:: path_list

    sys.stdin = stdin
    os.chdir(cwd)
    shutil.rmtree(tmpdir)

.. testcode:: path_list

    from jsonargparse.typing import Path_fr

    parser.add_argument("--list", type=List[Path_fr], enable_path=True)
    cfg = parser.parse_args(["--list", "paths.lst"])  # File with list of paths
    cfg = parser.parse_args(["--list", "-"])  # List of paths from stdin

If ``nargs='+'`` is given to ``add_argument`` with ``List[<path_type>]`` and
``enable_path=True`` then for each argument a list of paths is generated.

.. note::

    The :class:`.Path` class is currently not fully supported in windows.


.. _parsing-urls:

Parsing URLs
------------

The :func:`.path_type` function also supports URLs which after parsing, the
:py:meth:`.Path.get_content` method can be used to perform a GET request to the
corresponding URL and retrieve its content. For this to work the *requests*
python package is required. Alternatively, :func:`.path_type` can also be used
for `fsspec <https://filesystem-spec.readthedocs.io>`__ supported file systems.
The respective optional package(s) will be installed along with jsonargparse if
installed with the ``urls`` or ``fsspec`` extras require as explained in section
:ref:`installation`.

The ``'u'`` flag is used to parse URLs using requests and the flag ``'s'`` to
parse fsspec file systems. For example if it is desired that an argument can be
either a readable file or URL, the type would be created as ``Path_fur =
path_type('fur')``. If the value appears to be a URL, a HEAD request would be
triggered to check if it is accessible. To get the content of the parsed path,
without needing to care if it is a local file or a URL, the
:py:meth:`.Path.get_content` method Scan be used.

If you import ``from jsonargparse import set_config_read_mode`` and then run
``set_config_read_mode(urls_enabled=True)`` or
``set_config_read_mode(fsspec_enabled=True)``, the following functions and
classes will also support loading from URLs:
:py:meth:`.ArgumentParser.parse_path`, :py:meth:`.ArgumentParser.get_defaults`
(``default_config_files`` argument), :class:`.ActionConfigFile`,
:class:`.ActionJsonSchema`, :class:`.ActionJsonnet` and :class:`.ActionParser`.
This means that a tool that can receive a configuration file via
:class:`.ActionConfigFile` is able to get the content from a URL, thus something
like the following would work:

.. code-block:: bash

    my_tool.py --config http://example.com/config.yaml

.. note::

    Relative paths inside a remote path are parsed as remote. For example, for a
    relative path ``model/state_dict.pt`` found inside
    ``s3://bucket/config.yaml``, its parsed absolute path becomes
    ``s3://bucket/model/state_dict.pt``.


.. _boolean-arguments:

Booleans
--------

Parsing boolean arguments is very common, however, the original argparse only
has a limited support for them, via ``store_true`` and ``store_false``.
Furthermore unexperienced users might mistakenly use ``type=bool`` which would
not provide the intended behavior.

With jsonargparse adding an argument with ``type=bool`` the intended action is
implemented. If given as values ``{'yes', 'true'}`` or ``{'no', 'false'}`` the
corresponding parsed values would be ``True`` or ``False``. For example:

.. testsetup:: boolean

    parser = ArgumentParser()

.. doctest:: boolean

    >>> parser.add_argument("--op1", type=bool, default=False)  # doctest: +IGNORE_RESULT
    >>> parser.add_argument("--op2", type=bool, default=True)  # doctest: +IGNORE_RESULT
    >>> parser.parse_args(["--op1", "yes", "--op2", "false"])
    Namespace(op1=True, op2=False)

Sometimes it is also useful to define two paired options, one to set ``True``
and the other to set ``False``. The :class:`.ActionYesNo` class makes this
straightforward. A couple of examples would be:

.. testsetup:: yes_no

    parser = ArgumentParser()

.. testcode:: yes_no

    from jsonargparse import ActionYesNo

    # --opt1 for true and --no_opt1 for false.
    parser.add_argument("--op1", action=ActionYesNo)
    # --with-opt2 for true and --without-opt2 for false.
    parser.add_argument("--with-op2", action=ActionYesNo(yes_prefix="with-", no_prefix="without-"))

If the :class:`.ActionYesNo` class is used in conjunction with ``nargs='?'`` the
options can also be set by giving as value any of ``{'true', 'yes', 'false',
'no'}``.


.. _enums:

Enum arguments
--------------

Another case of restricted values is string choices. In addition to the common
``choices`` given as a list of strings, it is also possible to provide as type
an ``Enum`` class. This has the added benefit that strings are mapped to some
desired values. For example:

.. testsetup:: enum

    parser = ArgumentParser()

.. doctest:: enum

    >>> import enum
    >>> class MyEnum(enum.Enum):
    ...     choice1 = -1
    ...     choice2 = 0
    ...     choice3 = 1
    ...
    >>> parser.add_argument("--op", type=MyEnum)  # doctest: +IGNORE_RESULT
    >>> parser.parse_args(["--op=choice1"])
    Namespace(op=<MyEnum.choice1: -1>)


.. _list-append:

List append
-----------

As detailed before, arguments with ``List`` type are supported. By default when
specifying an argument value, the previous value is replaced, and this also
holds for lists. Thus, a parse such as ``parser.parse_args(['--list=[1]',
'--list=[2, 3]'])`` would result in a final value of ``[2, 3]``. However, in
some cases it might be decided to append to the list instead of replacing. This
can be achieved by adding ``+`` as suffix to the argument key, for example:

.. testsetup:: append

    parser = ArgumentParser()


    class MyBaseClass:
        pass

.. doctest:: append

    >>> parser.add_argument("--list", type=List[int])  # doctest: +IGNORE_RESULT
    >>> parser.parse_args(["--list=[1]", "--list+=[2, 3]"])
    Namespace(list=[1, 2, 3])
    >>> parser.parse_args(["--list=[4]", "--list+=5"])
    Namespace(list=[4, 5])

Append is also supported in config files. For instance the following two config
files would first assign a list and then append to this list:

.. code-block:: yaml

    # config1.yaml
    list:
    - 1

.. code-block:: yaml

    # config2.yaml
    list+:
    - 2
    - 3

Appending works for any type for the list elements. Lists with class type
elements (see :ref:`sub-classes`) are also supported. To append to the list,
first append a new class by using the ``+`` suffix. Then ``init_args`` for this
class are specified like if the type wasn't a list, since the arguments are
applied to the last class in the list. Take for example that an argument is
added to a parser as:

.. testcode:: append

    parser.add_argument("--list_of_instances", type=List[MyBaseClass])

Thanks to the short notation, command line arguments don't require to specify
``class_path`` and ``init_args``. Thus, multiple classes can be appended and its
arguments set as follows:

.. code-block:: bash

    python tool.py \
      --list_of_instances+={CLASS_1_PATH} \
      --list_of_instances.{CLASS_1_ARG_1}=... \
      --list_of_instances.{CLASS_1_ARG_2}=... \
      --list_of_instances+={CLASS_2_PATH} \
      --list_of_instances.{CLASS_2_ARG_1}=... \
      ...
      --list_of_instances+={CLASS_N_PATH} \
      --list_of_instances.{CLASS_N_ARG_1}=... \
      ...

Once a new class has been appended to the list, it is not possible to modify the
arguments of a previous class. This limitation is by intention since it forces
classes and its arguments to be defined in order, making the command line call
intuitive to write and understand.


.. _dict-items:

Dict items
----------

When an argument has ``Dict`` as type, the value can be set using json format,
e.g.:

.. testsetup:: dict_items

    parser = ArgumentParser()

.. doctest:: dict_items

    >>> parser.add_argument("--dict", type=dict)  # doctest: +IGNORE_RESULT
    >>> parser.parse_args(['--dict={"key1": "val1", "key2": "val2"}'])
    Namespace(dict={'key1': 'val1', 'key2': 'val2'})

Similar to lists, providing a second argument with value a json dict completely
replaces the previous value. Setting individual dict items without replacing can
be achieved as follows:

.. doctest:: dict_items

    >>> parser.parse_args(["--dict.key1=val1", "--dict.key2=val2"])
    Namespace(dict={'key1': 'val1', 'key2': 'val2'})


.. _dataclass-like:

Dataclass-like classes
----------------------

In contrast to subclasses, which requires the user to provide a ``class_path``,
in some cases it is not expected to have subclasses. In this case the init args
are given directly in a dictionary without specifying a ``class_path``. This is
the behavior for standard ``dataclasses``, ``final`` classes, attrs' ``define``
decorator, and pydantic's ``dataclass`` decorator and ``BaseModel`` classes.

As an example, take a class that is decorated with :func:`.final`, meaning that
it shouldn't be subclassed. The code below would accept the corresponding yaml
structure.

.. testsetup:: final_classes

    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="_jsonargparse_doctest_")
    os.chdir(tmpdir)
    with open("config.yaml", "w") as f:
        f.write("data:\n  number: 8\n  accepted: true\n")

.. testcleanup:: final_classes

    os.chdir(cwd)
    shutil.rmtree(tmpdir)

.. testcode:: final_classes

    from jsonargparse.typing import final


    @final
    class FinalClass:
        def __init__(self, number: int = 0, accepted: bool = False):
            ...


    parser = ArgumentParser()
    parser.add_argument("--data", type=FinalClass)
    cfg = parser.parse_path("config.yaml")

.. code-block:: yaml

    data:
      number: 8
      accepted: true


.. _callable-type:

Callable type
-------------

When using ``Callable`` as type, the parser accepts several options. The first
option is the import path of a callable object, for example:

.. testsetup:: callable

    parser = ArgumentParser()

.. testcode:: callable

    parser.add_argument("--callable", type=Callable)
    parser.parse_args(["--callable=time.sleep"])

A second option is a class that once instantiated becomes callable:

.. testcode:: callable

    class OffsetSum:
        def __init__(self, offset: int):
            self.offset = offset

        def __call__(self, value: int):
            return self.offset + value

.. testcode:: callable
    :hide:

    doctest_mock_class_in_main(OffsetSum)

.. doctest:: callable

    >>> value = {
    ...     "class_path": "__main__.OffsetSum",
    ...     "init_args": {
    ...         "offset": 3,
    ...     },
    ... }

    >>> cfg = parser.parse_args(["--callable", str(value)])
    >>> cfg.callable
    Namespace(class_path='__main__.OffsetSum', init_args=Namespace(offset=3))
    >>> init = parser.instantiate_classes(cfg)
    >>> init.callable(5)
    8

The third option is only applicable when the type is a callable that has a class
as return type or a ``Union`` including a class. This is useful to support
dependency injection for classes that require a parameter that is only available
after injection. The parser supports this automatically by providing a function
that receives this parameter and returns the instance of the class. Take for
example the classes:

.. testcode:: callable

    class Optimizer:
        def __init__(self, params: Iterable):
            self.params = params


    class SGD(Optimizer):
        def __init__(self, params: Iterable, lr: float):
            super().__init__(params)
            self.lr = lr

.. testcode:: callable
    :hide:

    doctest_mock_class_in_main(SGD)

A possible parser and callable behavior would be:

.. doctest:: callable

    >>> value = {
    ...     "class_path": "SGD",
    ...     "init_args": {
    ...         "lr": 0.01,
    ...     },
    ... }

    >>> parser.add_argument("--optimizer", type=Callable[[Iterable], Optimizer])  # doctest: +IGNORE_RESULT
    >>> cfg = parser.parse_args(["--optimizer", str(value)])
    >>> cfg.optimizer
    Namespace(class_path='__main__.SGD', init_args=Namespace(lr=0.01))
    >>> init = parser.instantiate_classes(cfg)
    >>> optimizer = init.optimizer([1, 2, 3])
    >>> isinstance(optimizer, SGD)
    True
    >>> optimizer.params, optimizer.lr
    ([1, 2, 3], 0.01)

.. note::

    When the ``Callable`` has a class return type, it is possible to specify the
    ``class_path`` giving only its name if imported before parsing, as explained
    in :ref:`sub-classes-command-line`.

If the same type above is used as type hint of a parameter of another class, a
default can be set using a lambda, for example:

.. testcode:: callable

    class Model:
        def __init__(
            self,
            optimizer: Callable[[Iterable], Optimizer] = lambda p: SGD(p, lr=0.05),
        ):
            self.optimizer = optimizer

Then a parser and behavior could be:

.. code-block::

    >>> parser.add_class_arguments(Model, 'model')
    >>> cfg = parser.get_defaults()
    >>> cfg.model.optimizer
    Namespace(class_path='__main__.SGD', init_args=Namespace(lr=0.05))
    >>> init = parser.instantiate_classes(cfg)
    >>> optimizer = init.model.optimizer([1, 2, 3])
    >>> optimizer.params, optimizer.lr
    ([1, 2, 3], 0.05)

See :ref:`ast-resolver` for limitations of lambda defaults.


.. _registering-types:

Registering types
-----------------

With the :func:`.register_type` function it is possible to register additional
types for use in jsonargparse parsers. If the type class can be instantiated
with a string representation and casting the instance to ``str`` gives back the
string representation, then only the type class is given to
:func:`.register_type`. For example in the ``jsonargparse.typing`` package this
is how complex numbers are registered: ``register_type(complex)``. For other
type classes that don't have these properties, to register it might be necessary
to provide a serializer and/or deserializer function. Including the serializer
and deserializer functions, the registration of the complex numbers example is
equivalent to ``register_type(complex, serializer=str, deserializer=complex)``.

A more useful example could be registering the ``datetime`` class. This case
requires to give both a serializer and a deserializer as seen below.

.. testcode::

    from datetime import datetime
    from jsonargparse import ArgumentParser
    from jsonargparse.typing import register_type


    def serializer(v):
        return v.isoformat()


    def deserializer(v):
        return datetime.strptime(v, "%Y-%m-%dT%H:%M:%S")


    register_type(datetime, serializer, deserializer)

    parser = ArgumentParser()
    parser.add_argument("--datetime", type=datetime)
    parser.parse_args(["--datetime=2008-09-03T20:56:35"])

.. note::

    The registering of types is only intended for simple types. By default any
    class used as a type hint is considered a sub-class (see :ref:`sub-classes`)
    which might be good for many use cases. If a class is registered with
    :func:`.register_type` then the sub-class option is no longer available.


.. _nested-namespaces:

Nested namespaces
=================

A difference with respect to basic argparse is, that by using dot notation in
the argument names, you can define a hierarchy of nested namespaces. For example
you could do the following:

.. doctest::

    >>> parser = ArgumentParser(prog="app")
    >>> parser.add_argument("--lev1.opt1", default="from default 1")  # doctest: +IGNORE_RESULT
    >>> parser.add_argument("--lev1.opt2", default="from default 2")  # doctest: +IGNORE_RESULT
    >>> cfg = parser.get_defaults()
    >>> cfg.lev1.opt1
    'from default 1'
    >>> cfg.lev1.opt2
    'from default 2'

A group of nested options can be created by using a dataclass. This has the
advantage that the same options can be reused in multiple places of a project.
An example analogous to the one above would be:

.. testcode::

    from dataclasses import dataclass


    @dataclass
    class Level1Options:
        """Level 1 options
        Args:
            opt1: Option 1
            opt2: Option 2
        """

        opt1: str = "from default 1"
        opt2: str = "from default 2"


    parser = ArgumentParser()
    parser.add_argument("--lev1", type=Level1Options, default=Level1Options())

The :class:`.Namespace` class is an extension of the one from argparse, having
some additional features. In particular, keys can be accessed like a dictionary
either with individual keys, e.g. ``cfg['lev1']['opt1']``, or a single one, e.g.
``cfg['lev1.opt1']``. Also the class has a method :py:meth:`.Namespace.as_dict`
that can be used to represent the nested namespace as a nested dictionary. This
is useful for example for class instantiation.


.. _configuration-files:

Configuration files
===================

An important feature of jsonargparse is the parsing of yaml/json files. The dot
notation hierarchy of the arguments (see :ref:`nested-namespaces`) are used for
the expected structure in the config files.

The :py:attr:`.ArgumentParser.default_config_files` property can be set when
creating a parser to specify patterns to search for configuration files. For
example if a parser is created as
``ArgumentParser(default_config_files=['~/.myapp.yaml', '/etc/myapp.yaml'])``,
when parsing if any of those two config files exist it will be parsed and used
to override the defaults. All matched config files are parsed and applied in the
given order. The default config files are always parsed first, this means that
any command line argument will override its values.

It is also possible to add an argument to explicitly provide a configuration
file path. Providing a config file as an argument does not disable the parsing
of ``default_config_files``. The config argument would be parsed in the specific
position among the command line arguments. Therefore the arguments found after
would override the values from that config file. The config argument can be
given multiple times, each overriding the values of the previous. Using the
example parser from the :ref:`nested-namespaces` section above, we could have
the following config file in yaml format:

.. code-block:: yaml

    # File: example.yaml
    lev1:
      opt1: from yaml 1
      opt2: from yaml 2

Then in python adding a config file argument and parsing some dummy arguments,
the following would be observed:

.. testsetup:: config

    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="_jsonargparse_doctest_")
    os.chdir(tmpdir)
    with open("example.yaml", "w") as f:
        f.write("lev1:\n  opt1: from yaml 1\n  opt2: from yaml 2\n")

.. testcleanup:: config

    os.chdir(cwd)
    shutil.rmtree(tmpdir)

.. doctest:: config

    >>> from jsonargparse import ArgumentParser, ActionConfigFile
    >>> parser = ArgumentParser()
    >>> parser.add_argument("--lev1.opt1", default="from default 1")  # doctest: +IGNORE_RESULT
    >>> parser.add_argument("--lev1.opt2", default="from default 2")  # doctest: +IGNORE_RESULT
    >>> parser.add_argument("--config", action=ActionConfigFile)  # doctest: +IGNORE_RESULT
    >>> cfg = parser.parse_args(["--lev1.opt1", "from arg 1", "--config", "example.yaml", "--lev1.opt2", "from arg 2"])
    >>> cfg.lev1.opt1
    'from yaml 1'
    >>> cfg.lev1.opt2
    'from arg 2'

Instead of providing a path to a configuration file, a string with the
configuration content can also be provided.

.. doctest:: config

    >>> cfg = parser.parse_args(["--config", '{"lev1":{"opt1":"from string 1"}}'])
    >>> cfg.lev1.opt1
    'from string 1'

The config file can also be provided as an environment variable as explained in
section :ref:`environment-variables`. The configuration file environment
variable is the first one to be parsed. Any other argument provided through an
environment variable would override the config file one.

A configuration file or string can also be parsed without parsing command line
arguments. The methods for this are :py:meth:`.ArgumentParser.parse_path` and
:py:meth:`.ArgumentParser.parse_string` to parse a config file or a config
string respectively.

Serialization
-------------

Parsers that have an :class:`.ActionConfigFile` argument also include a
``--print_config`` option. This is useful particularly for command line tools
with a large set of options to create an initial config file including all
default values. If the `ruyaml <https://ruyaml.readthedocs.io>`__ package is
installed, the config can be printed having the help descriptions content as
yaml comments by using ``--print_config=comments``. Another option is
``--print_config=skip_null`` which skips entries whose value is ``null``.

From within python it is also possible to serialize a config object by using
either the :py:meth:`.ArgumentParser.dump` or :py:meth:`.ArgumentParser.save`
methods. Three formats with a particular style are supported: ``yaml``, ``json``
and ``json_indented``. It is possible to add more dumping formats by using the
:func:`.set_dumper` function. For example to allow dumping using PyYAML's
``default_flow_style`` do the following:

.. testcode::

    import yaml
    from jsonargparse import set_dumper


    def custom_yaml_dump(data):
        return yaml.safe_dump(data, default_flow_style=True)


    set_dumper("yaml_custom", custom_yaml_dump)

.. _custom-loaders:

Custom loaders
--------------

The ``yaml`` parser mode (see :py:meth:`.ArgumentParser.__init__`) uses for
loading a subclass of `yaml.SafeLoader
<https://pyyaml.docsforge.com/master/api/yaml/loader/SafeLoader/>`__ with two
modifications. First, it supports float's scientific notation, e.g. ``'1e-3' =>
0.001`` (unlike default PyYAML which considers ``'1e-3'`` a string). Second,
text within curly braces is considered a string, e.g. ``'{text}' (unlike default
PyYAML which parses this as ``{'text': None}``).

It is possible to replace the yaml loader or add a loader as a new parser mode
via the :func:`.set_loader` function. For example if you need a custom PyYAML
loader it can be registered and used as follows:

.. testcode::

    import yaml
    from jsonargparse import ArgumentParser, set_loader


    class CustomLoader(yaml.SafeLoader):
        ...


    def custom_yaml_load(stream):
        return yaml.load(stream, Loader=CustomLoader)


    set_loader("yaml_custom", custom_yaml_load)

    parser = ArgumentParser(parser_mode="yaml_custom")

When setting a loader based on a library different from PyYAML, the ``exceptions``
that it raises when there are failures should be given to :func:`.set_loader`.


.. _classes-methods-functions:

Classes, methods and functions
==============================

It is good practice to write python code in which parameters have type hints and
these are described in the docstrings. To make this well written code
configurable, it wouldn't make sense to duplicate information of types and
parameter descriptions. To avoid this duplication, jsonargparse includes methods
to automatically add annotated parameters as arguments, see
:class:`.SignatureArguments`.

Take for example a class with its init and a method with docstrings as follows:

.. testsetup:: class_method

    sys.argv = ["", "--myclass.init.foo={}", "--myclass.method.bar=0"]


    class MyBaseClass:
        pass

.. testcode:: class_method

    from typing import Dict, Union, List


    class MyClass(MyBaseClass):
        def __init__(self, foo: Dict[str, Union[int, List[int]]], **kwargs):
            """Initializer for MyClass.

            Args:
                foo: Description for foo.
            """
            super().__init__(**kwargs)
            ...

        def mymethod(self, bar: float, baz: bool = False):
            """Description for mymethod.

            Args:
                bar: Description for bar.
                baz: Description for baz.
            """
            ...

Both ``MyClass`` and ``mymethod`` can easily be made configurable, the class
initialized and the method executed as follows:

.. testcode:: class_method

    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_class_arguments(MyClass, "myclass.init")
    parser.add_method_arguments(MyClass, "mymethod", "myclass.method")

    cfg = parser.parse_args()
    myclass = MyClass(**cfg.myclass.init.as_dict())
    myclass.mymethod(**cfg.myclass.method.as_dict())


The :func:`add_class_arguments` call adds to the ``myclass.init`` key the
``items`` argument with description as in the docstring, sets it as required
since it lacks a default value. When parsed, it is validated according to the
type hint, i.e., a dict with values ints or list of ints. Also since the init
has the ``**kwargs`` argument, the keyword arguments from ``MyBaseClass`` are
also added to the parser. Similarly, the :func:`add_method_arguments` call adds
to the ``myclass.method`` key, the arguments ``value`` as a required float and
``flag`` as an optional boolean with default value false.

Instantiation of several classes added with :func:`add_class_arguments` can be
done more simply for an entire config object using
:py:meth:`.ArgumentParser.instantiate_classes`. For the example above running
``cfg = parser.instantiate_classes(cfg)`` would result in ``cfg.myclass.init``
containing an instance of ``MyClass`` initialized with whatever command line
arguments were parsed.

When parsing from a configuration file (see :ref:`configuration-files`) all the
values can be given in a single config file. For convenience it is also possible
that the values for each of the argument groups created by the calls to add
signatures methods can be parsed from independent files. This means that for the
example above there could be one general config file with contents:

.. code-block:: yaml

    myclass:
      init: myclass.yaml
      method: mymethod.yaml

Then the files ``myclass.yaml`` and ``mymethod.yaml`` would include the settings
for the instantiation of the class and the call to the method respectively.

A wide range of type hints are supported for the signature parameters. For exact
details go to section :ref:`type-hints`. Some notes about the add signature
methods are:

- All positional only parameters must have a type, otherwise the add arguments
  functions raise an exception.

- Keyword parameters are ignored if they don't have at least one type that is
  supported.

- Parameters whose name starts with ``_`` are considered internal and ignored.

- The signature methods have a ``skip`` parameter which can be used to exclude
  adding some arguments, e.g. ``parser.add_method_arguments(MyClass, 'mymethod',
  skip={'flag'})``.

.. note::

    The signatures support is intended to be non-intrusive. It is by design that
    there is no need to inherit from a class, add decorators, or use special
    type hints and default values. This has several advantages. For example it
    is possible to use classes from third party libraries which is not possible
    for developers to modify.

Docstring parsing
-----------------

To get parameter docstrings in the parser help, the `docstring-parser
<https://pypi.org/project/docstring-parser/>`__ package is required. This
package is included when installing jsonargparse with the ``signatures`` extras
require as explained in section :ref:`installation`.

A couple of options can be configured, both related to docstring parsing speed.
By default docstrings are parsed used with
``docstring_parser.DocstringStyle.AUTO``, which means that it is attempted to
parse docstrings with all supported styles. If the relevant codebase uses a
single style, this is inefficient. A single style can be configured as follows:

.. testcode:: docstrings

    from docstring_parser import DocstringStyle
    from jsonargparse import set_docstring_parse_options

    set_docstring_parse_options(style=DocstringStyle.REST)

The second option that can be configured is the support for `attribute
docstrings <https://peps.python.org/pep-0257/#what-is-a-docstring>`__ (i.e.
literal strings in the line after an attribute is defined). By default this
feature is disabled and enabling it makes the parsing slower even for classes
that don't have attribute docstrings. To enable, do as follows:

.. testcode:: docstrings

    from dataclasses import dataclass
    from jsonargparse import set_docstring_parse_options

    set_docstring_parse_options(attribute_docstrings=True)


    @dataclass
    class Options:
        """Options for a competition winner."""

        name: str
        """Name of winner."""
        prize: int = 100
        """Amount won."""


.. testcleanup:: docstrings

    set_docstring_parse_options(style=DocstringStyle.GOOGLE)
    set_docstring_parse_options(attribute_docstrings=False)


Classes from functions
----------------------

In some cases there are functions which return an instance of a class. To add
this to a parser such that :py:meth:`.ArgumentParser.instantiate_classes` calls
this function, the example above would change to:

.. testsetup:: class_from_function

    class MyClass:
        pass


    def instantiate_myclass() -> MyClass:
        return MyClass()

.. testcode:: class_from_function

    from jsonargparse import ArgumentParser, class_from_function

    parser = ArgumentParser()
    dynamic_class = class_from_function(instantiate_myclass)
    parser.add_class_arguments(dynamic_class, "myclass.init")

.. note::

    :func:`.class_from_function` requires the input function to have a return
    type annotation that must be the class type it returns.

Classes created with :func:`.class_from_function` can be selected using
``class_path`` for :ref:`sub-classes`. For example, if
:func:`.class_from_function` is run in a module ``my_module`` as:

.. testcode:: class_from_function

    class_from_function(instantiate_myclass, name="MyClass")

Then the ``class_path`` for the created class would be ``my_module.MyClass``.


Parameter resolvers
-------------------

Three techniques are implemented for resolving signature parameters. One makes
use of python's `Abstract Syntax Trees (AST)
<https://docs.python.org/3/library/ast.html>`__ library and the second is based
on assumptions of class inheritance. The AST resolver is used first and only
when AST fails, the assumptions resolver is run as fallback. The third resolver
uses stub files ``*.pyi`` and is applied on top of both the AST and assumptions
resolvers.

Unresolved parameters
^^^^^^^^^^^^^^^^^^^^^

The parameter resolvers make a best effort to determine the correct names and
types that the parser should accept. However, there can be cases not yet
supported or cases for which it would be impossible to support. To somewhat
overcome these limitations, there is a special key ``dict_kwargs`` that can be
used to provide arguments that will not be validated during parsing, but will be
used for class instantiation. It is called ``dict_kwargs`` because there are use
cases in which ``**kwargs`` is used just as a dict, thus it also serves that
purpose.

Take for example the following parsing and instantiation:

.. testsetup:: unresolved

    sys.argv = ["", "--myclass=MyClass"]


    class MyClass:
        def __init__(self, foo: int = 0, **kwargs):
            super().__init__(**kwargs)
            ...


    MyClass.__module__ = "jsonargparse_tests"
    jsonargparse_tests.MyClass = MyClass

.. testcode:: unresolved

    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--myclass", type=MyClass)
    cfg = parser.parse_args()
    cfg_init = parser.instantiate_classes(cfg)

If ``MyClass.__init__`` has ``**kwargs`` with some unresolved parameters, the
following could be a valid config file:

.. code-block:: yaml

    class_path: MyClass
    init_args:
      foo: 1
    dict_kwargs:
      bar: 2

The value for ``bar`` will not be validated, but the class will be instantiated
as ``MyClass(foo=1, bar=2)``.

Assumptions resolver
^^^^^^^^^^^^^^^^^^^^

The assumptions resolver only considers classes. Whenever the ``__init__``
method has ``*args`` and/or ``**kwargs``, the resolver assumes that these are
directly forwarded to the next parent class, i.e. ``__init__`` includes a line
like ``super().__init__(*args, **kwargs)``. Thus, it blindly collects the
``__init__`` parameters of parent classes. The collected parameters will be
incorrect if the code does not follow this pattern. This is why it is only used
as fallback when the AST resolver fails.

.. _ast-resolver:

AST resolver
^^^^^^^^^^^^

The AST resolver analyzes the source code and tries to figure out how the
``*args`` and ``**kwargs`` are used to further find more accepted parameters.
This type of resolving is limited to a few specific cases since there are
endless possibilities for what code can do. The supported cases are illustrated
below. Bear in mind that the code does not need to be exactly like this. The
important detail is how ``*args`` and ``**kwargs`` are used, not other
parameters, or the names of variables, or the complexity of the code that is
unrelated to these variables.

.. testsetup:: ast_resolver

    class BaseClass:
        pass


    class SomeClass:
        def __init__(self, **kwargs):
            pass


    class ChildClass(BaseClass):
        def __init__(self, *args, **kwargs):
            pass

**Cases for statements in functions or methods**

.. testcode:: ast_resolver

    def calls_a_function(*args, **kwargs):
        a_function(*args, **kwargs)


    def calls_a_method(*args, **kwargs):
        an_instance = SomeClass()
        an_instance.a_method(*args, **kwargs)


    def calls_a_static_method(*args, **kwargs):
        an_instance = SomeClass()
        an_instance.a_static_method(*args, **kwargs)


    def calls_a_class_method(*args, **kwargs):
        SomeClass.a_class_method(*args, **kwargs)


    def pops_from_kwargs(**kwargs):
        val = kwargs.pop("name", "default")


    def gets_from_kwargs(**kwargs):
        val = kwargs.get("name", "default")


    def constant_conditional(**kwargs):
        if global_boolean_1:
            first_function(**kwargs)
        elif not global_boolean_2:
            second_function(**kwargs)
        else:
            third_function(**kwargs)

**Cases for classes**

.. testcode:: ast_resolver

    class PassThrough(BaseClass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


    class CallMethod:
        def __init__(self, *args, **kwargs):
            self.a_method(*args, **kwargs)


    class AttributeUseInMethod:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        def a_method(self):
            a_callable(**self._kwargs)


    class AttributeUseInProperty:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        @property
        def a_property(self):
            return a_callable(**self._kwargs)


    class DictUpdateUseInMethod:
        def __init__(self, **kwargs):
            self._kwargs = dict(p1=1)  # Can also be: self._kwargs = {'p1': 1}
            self._kwargs.update(**kwargs)  # Can also be: self._kwargs = dict(p1=1, **kwargs)

        def a_method(self):
            a_callable(**self._kwargs)


    class InstanceInClassmethod:
        @classmethod
        def get_instance(cls, **kwargs):
            return cls(**kwargs)


    class NonImmediateSuper(BaseClass):
        def __init__(self, *args, **kwargs):
            super(BaseClass, self).__init__(*args, **kwargs)

**Cases for class instance defaults**

.. testcode:: ast_resolver

    # Class instance: only keyword arguments with ``ast.Constant` value
    class_instance: SomeClass = SomeClass(param=1)

    # Lambda returning class instance: only keyword arguments with ``ast.Constant` value
    class_instance: Callable[[type], BaseClass] = lambda a: ChildClass(a, param=2.3)

There can be other parameters apart from ``*args`` and ``**kwargs``, thus in the
cases above, the signatures can be for example like ``name(p1: int, k1: str =
'a', **kws)``. Also when internally calling some function or instantiating a
class, there can be additional parameters. For example in:

.. testcode::

    def calls_a_function(*args, **kwargs):
        a_function(*args, param=1, **kwargs)

The ``param`` parameter would be excluded from the resolved parameters because
it is internally hard coded.

A special case which is supported but with caveats, is multiple calls that use
``**kwargs``. For example:

.. testcode:: ast_resolver

    def conditional_calls(**kwargs):
        if condition_1:
            first_function(**kwargs)
        elif condition_2:
            second_function(**kwargs)
        else:
            third_function(**kwargs)

The resolved parameters that have the same type hint and default accross all
calls are supported normally. When there is a discrepancy between the calls, the
parameters behave differently and are shown in the help in special "Conditional
arguments" sections. The main difference is that these arguments are not
included in :py:meth:`.ArgumentParser.get_defaults` or the output of
``--print_config``. This is necessary because the parser does not know which of
the calls will be used at runtime, and adding them would cause
:py:meth:`.ArgumentParser.instantiate_classes` to fail due to unexpected keyword
arguments.

.. note::

    The parameter resolvers log messages of failures and unsupported cases. To
    view these logs, set the environment variable ``JSONARGPARSE_DEBUG`` to any
    value. The supported cases are limited and it is highly encouraged that
    people create issues requesting the support for new ones. However, note that
    when a case is highly convoluted it could be a symptom that the respective
    code is in need of refactoring.

.. _stubs-resolver:

Stubs resolver
^^^^^^^^^^^^^^

The stubs resolver makes use of the `typeshed-client
<https://pypi.org/project/typeshed-client/>`__ package to identify parameters
and their type hints from stub files ``*.pyi``. To enable this resolver, install
jsonargparse with the ``signatures`` extras require as explained in section
:ref:`installation`.

Many of the types defined in stub files use the latest syntax for type hints,
that is, bitwise or operator ``|`` for unions and generics, e.g.
``list[<type>]`` instead of ``typing.List[<type>]``, see PEPs `604
<https://peps.python.org/pep-0604>`__ and `585
<https://peps.python.org/pep-0585>`__. On python>=3.10 these are fully
supported. On python<=3.9 backporting these types is attempted and in some cases
it can fail. On failure the type annotation is set to ``Any``.

Most of the types in the Python standard library have their types in stubs. An
example from the standard library would be:

.. doctest:: stubs_resolver

    >>> from random import uniform

    >>> parser = ArgumentParser()
    >>> parser.add_function_arguments(uniform, "uniform")  # doctest: +IGNORE_RESULT
    >>> parser.parse_args(["--uniform.a=0.7", "--uniform.b=3.4"])
    Namespace(uniform=Namespace(a=0.7, b=3.4))

Without the stubs resolver, the
:py:meth:`.SignatureArguments.add_function_arguments` call requires the
``fail_untyped=False`` option. This has the disadvantage that type ``Any`` is
given to the ``a`` and ``b`` arguments, instead of ``float``. And this means
that the parser would not fail if given an invalid value, for instance a string.


.. _sub-classes:

Class type and sub-classes
==========================

It is also possible to use an arbitrary class as a type such that the argument
accepts this class or any derived subclass. In the config file a class is
represented by a dictionary with a ``class_path`` entry indicating the dot
notation expression to import the class, and optionally some ``init_args`` that
would be used to instantiate it. When parsing, it will be checked that the class
can be imported, that it is a subclass of the given type and that ``init_args``
values correspond to valid arguments to instantiate it. After parsing, the
config object will include the ``class_path`` and ``init_args`` entries. To get
a config object with all sub-classes instantiated, the
:py:meth:`.ArgumentParser.instantiate_classes` method is used. The ``skip``
parameter of the signature methods can also be used to exclude arguments within
subclasses. This is done by giving its relative destination key, i.e. as
``param.init_args.subparam``.

A simple example would be having some config file ``config.yaml`` as:

.. code-block:: yaml

    myclass:
      calendar:
        class_path: calendar.Calendar
        init_args:
          firstweekday: 1

Then in python:

.. testsetup:: subclasses

    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="_jsonargparse_doctest_")
    os.chdir(tmpdir)
    with open("config.yaml", "w") as f:
        f.write("myclass:\n  calendar:\n    class_path: calendar.Calendar\n    init_args:\n      firstweekday: 1\n")

.. testcleanup:: subclasses

    os.chdir(cwd)
    shutil.rmtree(tmpdir)

.. doctest:: subclasses

    >>> from calendar import Calendar

    >>> class MyClass:
    ...     def __init__(self, calendar: Calendar):
    ...         self.calendar = calendar
    ...

    >>> parser = ArgumentParser()
    >>> parser.add_class_arguments(MyClass, "myclass")  # doctest: +IGNORE_RESULT

    >>> cfg = parser.parse_path("config.yaml")
    >>> cfg.myclass.calendar.as_dict()
    {'class_path': 'calendar.Calendar', 'init_args': {'firstweekday': 1}}

    >>> cfg = parser.instantiate_classes(cfg)
    >>> cfg.myclass.calendar.getfirstweekday()
    1

In this example the ``class_path`` points to the same class used for the type.
But a subclass of ``Calendar`` with an extended set of init parameters would
also work.

An individual argument can also be added having as type a class, i.e.
``parser.add_argument('--calendar', type=Calendar)``. There is also another
method :py:meth:`.SignatureArguments.add_subclass_arguments` which does the same
as ``add_argument``, but has some added benefits: 1) the argument is added in a
new group automatically; 2) the argument values can be given in an independent
config file by specifying a path to it; and 3) by default sets a useful
``metavar`` and ``help`` strings.

.. note::

    Classes will be parsed and instantiated when given as value a dict with
    ``class_path`` and ``init_args`` if the corresponding parameter has type
    ``Any``, or when ``fail_untyped=False`` which defaults to type ``Any``.

.. _sub-classes-command-line:

Command line
------------

The help of the parser does not show details for a type class since this depends
on the subclass. To get details for a particular subclass there is a specific
help option that receives the import path. Take for example a parser defined as:

.. testcode::

    from calendar import Calendar
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--calendar", type=Calendar)

The help for a corresponding subclass could be printed as:

.. code-block:: bash

    python tool.py --calendar.help calendar.TextCalendar

In the command line, a subclass can be specified through multiple command line
arguments:

.. code-block:: bash

    python tool.py \
      --calendar.class_path calendar.TextCalendar \
      --calendar.init_args.firstweekday 1

For convenience, the arguments can be somewhat shorter by omitting
``.class_path`` and ``.init_args`` and only specifying the name of the subclass
instead of the full import path.

.. code-block:: bash

    python tool.py --calendar TextCalendar --calendar.firstweekday 1

Specifying the name of the subclass works for subclasses in modules that have
been imported before parsing. Abstract classes and private classes (module or
name starting with ``'_'``) are not considered. All the subclasses resolvable by
its name can be seen in the general help ``python tool.py --help``.

Default values
--------------

For a parameter that has a class as type, it might also be wanted to set a
default value for it. Special care must be taken when doing this, could be
considered bad practice and be a good idea to avoid in most cases. The issue is
that classes are normally mutable. Depending on how the parameter value is used,
its default class instance in the signature could be changed. This goes against
what a default value is expected to be and lead to bugs which are difficult to
debug.

Since there are some legitimate use cases for class instances in defaults, they
are supported with a particular behavior and recommendations. An example is:

.. testcode:: instance_default

    class MyClass:
        def __init__(
            self,
            calendar: Calendar = Calendar(firstweekday=1),
        ):
            self.calendar = calendar

Adding this class to a parser will work without issues. The :ref:`ast-resolver`
in limited cases determines how to instantiate the original default. The parsing
methods would provide a dict with ``class_path`` and ``init_args`` instead of
the class instance. Furthermore, if
:py:meth:`.ArgumentParser.instantiate_classes` is used, a new instance of the
class is created, thereby avoiding issues related to the mutability of the
default.

Since the :ref:`ast-resolver` only supports limited cases, or when the source
code is not available, a second approach is to use the special function
:func:`.lazy_instance` to instantiate the default. Continuing with the same
example above, this would be:

.. testcode:: instance_default

    from jsonargparse import lazy_instance


    class MyClass:
        def __init__(
            self,
            calendar: Calendar = lazy_instance(Calendar, firstweekday=1),
        ):
            self.calendar = calendar

Like this, the parsed default will be a dict with ``class_path`` and
``init_args``, again avoiding the risk of mutability.

.. note::

    In python there can be some classes or functions for which it is not
    possible to determine its import path from the object alone. When using one
    of these as a default would cause a failure when serializing because what
    gets saved in the config file is the import path. To overcome this problem
    use the :func:`.register_unresolvable_import_paths` function giving it the
    module from where the respective object can be imported.


.. _argument-linking:

Argument linking
================

Some use cases could require adding arguments from multiple classes and some
parameters get a value automatically computed from other arguments. This
behavior can be obtained by using the :py:meth:`.ArgumentLinking.link_arguments`
method.

There are two types of links, defined with ``apply_on='parse'`` or
``apply_on='instantiate'``. As the names suggest, the former are set when
calling one of the parse methods and the latter are set when calling
:py:meth:`.ArgumentParser.instantiate_classes`.

For parsing links, source keys can be individual arguments or nested groups. The
target key has to be a single argument. The keys can be inside init_args of a
subclass. The compute function should accept as many positional arguments as
there are sources and return a value of type compatible with the target. An
example would be the following:

.. testcode::

    class Model:
        def __init__(self, batch_size: int):
            self.batch_size = batch_size


    class Data:
        def __init__(self, batch_size: int = 5):
            self.batch_size = batch_size


    parser = ArgumentParser()
    parser.add_class_arguments(Model, "model")
    parser.add_class_arguments(Data, "data")
    parser.link_arguments("data.batch_size", "model.batch_size", apply_on="parse")

As argument and in config files only ``data.batch_size`` should be specified.
Then whatever value it has will be propagated to ``model.batch_size``.

For instantiation links, sources can be class groups (added with
:py:meth:`.SignatureArguments.add_class_arguments`) or subclass arguments (see
:ref:`sub-classes`). The source key can be the entire instantiated object or an
attribute of the object. The target key has to be a single argument and can be
inside init_args of a subclass. The order of instantiation used by
:py:meth:`.ArgumentParser.instantiate_classes` is automatically determined based
on the links. The set of all instantiation links must be a directed acyclic
graph. An example would be the following:

.. testcode::

    class Model:
        def __init__(self, num_classes: int):
            self.num_classes = num_classes


    class Data:
        def __init__(self):
            self.num_classes = get_num_classes()


    parser = ArgumentParser()
    parser.add_class_arguments(Model, "model")
    parser.add_class_arguments(Data, "data")
    parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")

This link would imply that :py:meth:`.ArgumentParser.instantiate_classes`
instantiates :class:`Data` first, then use the ``num_classes`` attribute to
instantiate :class:`Model`.


Variable interpolation
======================

One of the possible reasons to add a parser mode (see :ref:`custom-loaders`) can
be to have support for variable interpolation in yaml files. Any library could
be used to implement a loader and configure a mode for it. Without needing to
implement a loader function, an ``omegaconf`` parser mode is available out of
the box when this package is installed.

Take for example a yaml file as:

.. code-block:: yaml

    server:
      host: localhost
      port: 80
    client:
      url: http://${server.host}:${server.port}/

.. testsetup:: omegaconf

    example = """
    server:
      host: localhost
      port: 80
    client:
      url: http://${server.host}:${server.port}/
    """
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="_jsonargparse_doctest_")
    os.chdir(tmpdir)
    with open("example.yaml", "w") as f:
        f.write(example)

.. testcleanup:: omegaconf

    os.chdir(cwd)
    shutil.rmtree(tmpdir)

This yaml could be parsed as follows:

.. doctest:: omegaconf

    >>> @dataclass
    ... class ServerOptions:
    ...     host: str
    ...     port: int
    ...

    >>> @dataclass
    ... class ClientOptions:
    ...     url: str
    ...

    >>> parser = ArgumentParser(parser_mode="omegaconf")
    >>> parser.add_argument("--server", type=ServerOptions)  # doctest: +IGNORE_RESULT
    >>> parser.add_argument("--client", type=ClientOptions)  # doctest: +IGNORE_RESULT
    >>> parser.add_argument("--config", action=ActionConfigFile)  # doctest: +IGNORE_RESULT

    >>> cfg = parser.parse_args(["--config=example.yaml"])
    >>> cfg.client.url
    'http://localhost:80/'

.. note::

    The ``parser_mode='omegaconf'`` provides support for `OmegaConf's
    <https://omegaconf.readthedocs.io/>`__ variable interpolation in a single
    yaml file. It is not possible to do interpolation across multiple yaml files
    or in an isolated individual command line argument.


.. _environment-variables:

Environment variables
=====================

The jsonargparse parsers can also get values from environment variables. The
parser checks existing environment variables whose name is of the form
``[PREFIX_][LEV__]*OPT``, that is, all in upper case, first a prefix (set by
``env_prefix``, or if unset the ``prog`` without extension or none if set to False)
followed by underscore and then the argument name replacing dots with two underscores.
Using the parser from the :ref:`nested-namespaces` section above, in your shell you
would set the environment variables as:

.. code-block:: bash

    export APP_LEV1__OPT1='from env 1'
    export APP_LEV1__OPT2='from env 2'

Then in python the parser would use these variables, unless overridden by the
command line arguments, that is:

.. testsetup:: env

    os.environ["APP_LEV1__OPT1"] = "from env 1"
    os.environ["APP_LEV1__OPT2"] = "from env 2"

.. doctest:: env

    >>> parser = ArgumentParser(env_prefix="APP", default_env=True)
    >>> parser.add_argument("--lev1.opt1", default="from default 1")  # doctest: +IGNORE_RESULT
    >>> parser.add_argument("--lev1.opt2", default="from default 2")  # doctest: +IGNORE_RESULT
    >>> cfg = parser.parse_args(["--lev1.opt1", "from arg 1"])
    >>> cfg.lev1.opt1
    'from arg 1'
    >>> cfg.lev1.opt2
    'from env 2'

Note that when creating the parser, ``default_env=True`` was given. By default
:py:meth:`.ArgumentParser.parse_args` does not parse environment variables. If
``default_env`` is left unset, environment variable parsing can also be enabled
by setting in your shell ``JSONARGPARSE_DEFAULT_ENV=true``.

There is also the :py:meth:`.ArgumentParser.parse_env` function to only parse
environment variables, which might be useful for some use cases in which there
is no command line call involved.

If a parser includes an :class:`.ActionConfigFile` argument, then the
environment variable for this config file will be parsed before all the other
environment variables.


.. _sub-commands:

Sub-commands
============

A way to define parsers in a modular way is what in argparse is known as
`sub-commands <https://docs.python.org/3/library/argparse.html#sub-commands>`__.
However, to promote modularity, in jsonargparse sub-commands work a bit
different than in argparse. To add sub-commands to a parser, the
:py:meth:`.ArgumentParser.add_subcommands` method is used. Then an existing
parser is added as a sub-command using :func:`.add_subcommand`. In a parsed
config object the sub-command will be stored in the ``subcommand`` entry (or
whatever ``dest`` was set to), and the values of the sub-command will be in an
entry with the same name as the respective sub-command. An example of defining a
parser with sub-commands is the following:

.. testcode::

    from jsonargparse import ArgumentParser

    ...
    parser_subcomm1 = ArgumentParser()
    parser_subcomm1.add_argument("--op1")
    ...
    parser_subcomm2 = ArgumentParser()
    parser_subcomm2.add_argument("--op2")
    ...
    parser = ArgumentParser(prog="app")
    parser.add_argument("--op0")
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("subcomm1", parser_subcomm1)
    subcommands.add_subcommand("subcomm2", parser_subcomm2)

Then some examples of parsing are the following:

.. doctest::

    >>> parser.parse_args(["subcomm1", "--op1", "val1"])  # doctest: +IGNORE_RESULT
    Namespace(op0=None, subcommand='subcomm1', subcomm1=Namespace(op1='val1'))
    >>> parser.parse_args(["--op0", "val0", "subcomm2", "--op2", "val2"])  # doctest: +IGNORE_RESULT
    Namespace(op0='val0', subcommand='subcomm2', subcomm2=Namespace(op2='val2'))

Parsing config files with :py:meth:`.ArgumentParser.parse_path` or
:py:meth:`.ArgumentParser.parse_string` is also possible. The config file is not
required to specify a value for ``subcommand``. For the example parser above a
valid yaml would be:

.. code-block:: yaml

    # File: example.yaml
    op0: val0
    subcomm1:
      op1: val1

Parsing of environment variables works similar to :class:`.ActionParser`. For
the example parser above, all environment variables for ``subcomm1`` would have
as prefix ``APP_SUBCOMM1_`` and likewise for ``subcomm2`` as prefix
``APP_SUBCOMM2_``. The sub-command to use could be chosen by setting environment
variable ``APP_SUBCOMMAND``.

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
this functionality install with the ``jsonschema`` extras require as explained
in section :ref:`installation`.

Check out the `jsonschema documentation
<https://python-jsonschema.readthedocs.io/>`__ to learn how to write a schema.
The current version of jsonargparse uses Draft7Validator. Parsing an argument
using a json schema is done like in the following example:

.. doctest::

    >>> from jsonargparse import ActionJsonSchema

    >>> schema = {
    ...     "type": "object",
    ...     "properties": {
    ...         "price": {"type": "number"},
    ...         "name": {"type": "string"},
    ...     },
    ... }

    >>> parser = ArgumentParser()
    >>> parser.add_argument("--json", action=ActionJsonSchema(schema=schema))  # doctest: +IGNORE_RESULT

    >>> parser.parse_args(["--json", '{"price": 1.5, "name": "cookie"}'])
    Namespace(json={'price': 1.5, 'name': 'cookie'})

Instead of giving a json string as argument value, it is also possible to
provide a path to a json/yaml file, which would be loaded and validated against
the schema. If the schema defines default values, these will be used by the
parser to initialize the config values that are not specified. When adding an
argument with the :class:`.ActionJsonSchema` action, you can use "%s" in the
``help`` string so that in that position the schema is printed.


.. _jsonnet-files:

Jsonnet files
=============

The Jsonnet support requires `jsonschema
<https://pypi.org/project/jsonschema/>`__ and `jsonnet
<https://pypi.org/project/jsonnet/>`__ python packages which are not included
with minimal jsonargparse install. To enable this functionality install
jsonargparse with the ``jsonnet`` extras require as explained in section
:ref:`installation`.

By default an :class:`.ArgumentParser` parses configuration files as yaml.
However, if instantiated giving ``parser_mode='jsonnet'``, then
:func:`parse_args`, :func:`parse_path` and :func:`parse_string` will expect
config files to be in jsonnet format instead. Example:

.. testsetup:: jsonnet

    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="_jsonargparse_doctest_")
    os.chdir(tmpdir)
    with open("example.jsonnet", "w") as f:
        f.write("{}\n")

.. testcleanup:: jsonnet

    os.chdir(cwd)
    shutil.rmtree(tmpdir)

.. testcode:: jsonnet

    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(parser_mode="jsonnet")
    parser.add_argument("--config", action=ActionConfigFile)
    cfg = parser.parse_args(["--config", "example.jsonnet"])

Jsonnet files are commonly parametrized, thus requiring external variables for
parsing. For these cases, instead of changing the parser mode away from yaml,
the :class:`.ActionJsonnet` class can be used. This action allows to define an
argument which would be a jsonnet string or a path to a jsonnet file. Moreover,
another argument can be specified as the source for any external variables
required, which would be either a path to or a string containing a json
dictionary of variables. Its use would be as follows:

.. testcode:: jsonnet

    from jsonargparse import ArgumentParser, ActionJsonnet, ActionJsonnetExtVars

    parser = ArgumentParser()
    parser.add_argument("--in_ext_vars", action=ActionJsonnetExtVars())
    parser.add_argument("--in_jsonnet", action=ActionJsonnet(ext_vars="in_ext_vars"))

For example, if a jsonnet file required some external variable ``param``, then
the jsonnet and the external variable could be given as:

.. testcode:: jsonnet

    cfg = parser.parse_args(["--in_ext_vars", '{"param": 123}', "--in_jsonnet", "example.jsonnet"])

Note that the external variables argument must be provided before the jsonnet
path so that this dictionary already exists when parsing the jsonnet.

The :class:`.ActionJsonnet` class also accepts as argument a json schema, in
which case the jsonnet would be validated against this schema right after
parsing.


.. _parser-arguments:

Parsers as arguments
====================

Sometimes it is useful to take an already existing parser that is required
standalone in some part of the code, and reuse it to parse an inner node of
another more complex parser. For these cases an argument can be defined using
the :class:`.ActionParser` class. An example of how to use this class is the
following:

.. testcode::

    from jsonargparse import ArgumentParser, ActionParser

    inner_parser = ArgumentParser(prog="app1")
    inner_parser.add_argument("--op1")
    ...
    outer_parser = ArgumentParser(prog="app2")
    outer_parser.add_argument("--inner.node", title="Inner node title", action=ActionParser(parser=inner_parser))

When using the :class:`.ActionParser` class, the value of the node in a config
file can be either the complex node itself, or the path to a file which will be
loaded and parsed with the corresponding inner parser. Naturally using
:class:`.ActionConfigFile` to parse a complete config file will parse the inner
nodes correctly.

Note that when adding ``inner_parser`` a title was given. In the help, the added
parsers are shown as independent groups starting with the given ``title``. It is
also possible to provide a ``description``.

Regarding environment variables, the prefix of the outer parser will be used to
populate the leaf nodes of the inner parser. In the example above, if
``inner_parser`` is used to parse environment variables, then as normal
``APP1_OP1`` would be checked to populate option ``op1``. But if
``outer_parser`` is used, then ``APP2_INNER__NODE__OP1`` would be checked to
populate ``inner.node.op1``.

An important detail to note is that the parsers that are given to
:class:`.ActionParser` are internally modified. Therefore, to use the parser
both as standalone and as inner node, it is necessary to implement a function
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
or by installing jsonargparse with the ``argcomplete`` extras require as
explained in section :ref:`installation`. Then the tab completion can be enabled
`globally <https://kislyuk.github.io/argcomplete/#global-completion>`__ for all
argcomplete compatible tools or for each `individual
<https://kislyuk.github.io/argcomplete/#synopsis>`__ tool. A simple
``example.py`` tool would be:

.. testsetup:: tab_completion

    sys.argv = [""]

.. testcode:: tab_completion

    #!/usr/bin/env python3

    from typing import Optional
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--bool", type=Optional[bool])

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

Troubleshooting and logging
===========================

The standard behavior for the parse methods, when they fail, is to print a short
message and terminate the process with a non-zero exit code. This is problematic
during development since there is not enough information to track down the root
of the problem. Without the need to change the source code, this default
behavior can be changed such that in case of failure, a ParseError exception is
raised and the full stack trace is printed. This is done by setting the
``JSONARGPARSE_DEBUG`` environment variable to any value.

The parsers from jsonargparse log some basic events, though by default this is
disabled. To enable, the ``logger`` argument should be set when creating an
:class:`.ArgumentParser` object. The intended use is to give as value an already
existing logger object which is used for the whole application. For convenience,
to enable a default logger the ``logger`` argument can also receive ``True`` or
a string which sets the name of the logger or a dictionary that can include the
name and the level, e.g. ``{"name": "myapp", "level": "ERROR"}``. If
`reconplogger <https://pypi.org/project/reconplogger/>`__ is installed, setting
``logger`` to ``True`` or a dictionary without specifying a name, then the
reconplogger is used. If reconplogger is installed and the
``JSONARGPARSE_DEBUG`` environment variable is set, then the logging level
becomes ``DEBUG``.
