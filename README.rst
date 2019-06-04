.. image:: https://circleci.com/gh/omni-us/yamlargparse.svg?style=svg
    :target: https://circleci.com/gh/omni-us/yamlargparse
.. image:: https://badge.fury.io/py/yamlargparse.svg
    :target: https://badge.fury.io/py/yamlargparse
.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg
    :target: https://github.com/omni-us/yamlargparse


yamlargparse python module
==========================

https://omni-us.github.io/yamlargparse/

This module is an extension to python's argparse which simplifies parsing of
configuration options from command line arguments, yaml configuration files,
environment variables and hard-coded defaults.

The aim is similar to other projects such as `configargparse
<https://pypi.org/project/ConfigArgParse/>`_, `yconf
<https://pypi.org/project/yconf/>`_ and `confuse
<https://pypi.org/project/confuse/>`_. The obvious question is, why yet another
module similar to many already existing ones? The answer is simply that none of
the existing projects had the exact features we wanted and after analyzing the
alternatives it seemed simpler to create a new module.


Features
========

- Parsers are configured just like with python's argparse, thus it has a gentile learning curve.

- Not exclusively intended for parsing command line arguments. The main focus is parsing yaml configuration files and not necessarily from a command line tool.

- Support for nested namespaces which makes it possible to parse yaml with non-flat hierarchies.

- Parsing of relative paths within yaml files and path lists.

- Default behavior is not identical to argparse, though it is possible to configure it to be identical. The main differences are:

  - When parsing fails :class:`ParserError` is raised, instead of printing usage and program exit.
  - To modify the behavior for parsing errors (e.g. print usage) an error handler function can be provided.

- Configuration settings are overridden based on the following precedence.

  - **Parsing command line:** command line arguments (might include config file) > environment variables > default config file > defaults.
  - **Parsing yaml:** config file > environment variables > default config file > defaults.
  - **Parsing environment:** environment variables > default config file > defaults.


Using the module
================

A parser is created just like it is done with argparse. You import the module,
create a parser object and then add arguments to it. A simple example would be:

.. code-block:: python

    import yamlargparse
    parser = yamlargparse.ArgumentParser(
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
the :func:`yamlargparse.ArgumentParser.parse_args` function, after which you get
an object with the parsed values or defaults available as attributes. For
illustrative purposes giving to :func:`parse_args` a list of arguments (instead
of automatically getting them from the command line arguments), with the parser
from above you would observe:

.. code-block:: python

    >>> cfg = parser.parse_args(['--opt2', '2.3'])
    >>> cfg.opt1, type(cfg.opt1)
    (0, <class 'int'>)
    >>> cfg.opt2, type(cfg.opt2)
    (2.3, <class 'float'>)

If the parsing fails a :class:`ParserError` is raised, so depending on the use case it
might be necessary to catch it.

.. code-block:: python

    >>> try:
    ...     cfg = parser.parse_args(['--opt2', 'four'])
    ... except yamlargparse.ParserError as ex:
    ...     print('parser error: '+str(ex))
    ...
    parser error: argument --opt2: invalid float value: 'four'

To get the default behavior of argparse the ArgumentParser can be initialized as
follows:

.. code-block:: python

    parser = yamlargparse.ArgumentParser(
        prog='app',
        error_handler=yamlargparse.usage_and_exit_error_handler,
        description='Description for my app.')


.. _nested-namespaces:

Nested namespaces
=================

A difference with respect to the basic argparse is that it by using dot notation
in the argument names, you can define a hierarchy of nested namespaces. So for
example you could do the following:

.. code-block:: python

    >>> parser = yamlargparse.ArgumentParser(prog='app')
    >>> parser.add_argument('--lev1.opt1', default='from default 1')
    >>> parser.add_argument('--lev1.opt2', default='from default 2')
    >>> cfg = parser.get_defaults()
    >>> cfg.lev1.opt1
    'from default 2'
    >>> cfg.lev1.opt2
    'from default 2'


Environment variables
=====================

The yamlargparse parsers by default also get values from environment variables.
The parser checks existing environment variables whose name is of the form
:code:`[PROG_][LEV__]*OPT`, that is all in upper case, first the name of the
program followed by underscore and then the argument name replacing dots with
two underscores. Using the parser from the :ref:`nested-namespaces` section
above, in your shell you would set the environment variables as:

.. code-block:: bash

    export APP_LEV1__OPT1='from env 1'
    export APP_LEV1__OPT2='from env 2'

Then in python the parser would use these variables, unless overridden by the
command line arguments, that is:

.. code-block:: python

    >>> parser = yamlargparse.ArgumentParser(prog='app')
    >>> parser.add_argument('--lev1.opt1', default='from default 1')
    >>> parser.add_argument('--lev1.opt2', default='from default 2')
    >>> cfg = parser.parse_args(['--lev1.opt1', 'from arg 1'])
    >>> cfg.lev1.opt1
    'from arg 1'
    >>> cfg.lev1.opt2
    'from env 2'

There is also the :func:`yamlargparse.ArgumentParser.parse_env` function to only
parse environment variables, which might be useful for some use cases in which
there is no command line call involved.


YAML configuration files
========================

An important feature of this module is the parsing of yaml files. The dot
notation hierarchy of the arguments (see :ref:`nested-namespaces`) are
used for the expected structure of the yaml files.

When creating the :class:`.ArgumentParser` the :code:`default_config_files`
argument can be given to specify patterns to search for configuration files.
Only the first matched config file is parsed.

When parsing command line arguments, it is possible to add a yaml configuration
file path argument. The yaml file would be read and parsed in the specific
position among the command line arguments, so the arguments after would override
the values from the yaml file. Again using the parser from the
:ref:`nested-namespaces` section above, for example we could have the
following yaml:

.. code-block:: yaml

    # File: example.yaml
    lev1:
      opt1: from yaml 1
      opt2: from yaml 2

Then in python adding a yaml file argument and parsing some example arguments,
the following would be observed:

.. code-block:: python

    >>> parser = yamlargparse.ArgumentParser(prog='app')
    >>> parser.add_argument('--lev1.opt1', default='from default 1')
    >>> parser.add_argument('--lev1.opt2', default='from default 2')
    >>> parser.add_argument('--cfg', action=yamlargparse.ActionConfigFile)
    >>> cfg = parser.parse_args(['--lev1.opt1', 'from arg 1', '--cfg', 'example.yaml', '--lev1.opt2', 'from arg 2'])
    >>> cfg.lev1.opt1
    'from yaml 1'
    >>> cfg.lev1.opt2
    'from arg 2'

There are also functions :func:`yamlargparse.ArgumentParser.parse_yaml_path` and
:func:`yamlargparse.ArgumentParser.parse_yaml_string` to only parse a yaml file
or yaml contained in a string respectively.


Parsing paths
=============

For some use cases it is necessary to parse file paths, checking its existence
and access permissions, but not necessarily opening the file. Moreover, a file
path could be included in a yaml file as relative with respect to the yaml
file's location. After parsing it should be easy to access the parsed file path
without having to consider the location of the yaml file. To help in these
situations yamlargparse includes the :class:`.ActionPath` and the
:class:`.ActionPathList` classes.

For example suppose you have a directory with a configuration file
:code:`app/config.yaml` and some data :code:`app/data/info.db`. The contents of
the yaml file is the following:

.. code-block:: yaml

    # File: config.yaml
    databases:
      info: data/info.db

To create a parser that checks that the value of :code:`databases.info` exists
and is readable, the following could be done:

.. code-block:: python

    >>> parser = yamlargparse.ArgumentParser(prog='app')
    >>> parser.add_argument('--databases.info', action=yamlargparse.ActionPath(mode='fr'))
    >>> cfg = parser.parse_yaml('app/config.yaml')

After parsing it is possible to get both the original relative path as included
in the yaml file, or the corresponding absolute path:

.. code-block:: python

    >>> cfg.databases.info(absolute=False)
    'data/info.db'
    >>> cfg.databases.info()
    '/YOUR_CWD/app/data/info.db'

Likewise directories can also be parsed by including in the mode the :code:`'d'`
flag, e.g. :code:`ActionPath(mode='drw')`.

An argument with :class:`.ActionPath` can be given :code:`nargs='+'` to parse
multiple paths. But it might also be wanted to parse a list of paths found in a
plain text file or from stdin. For this the :class:`.ActionPathList` is used and
as argument either the path to a file listing the paths is given or the special
:code:`'-'` string for reading the list from stdin. For for example:

.. code-block:: python

    >>> parser.add_argument('--list', action=yamlargparse.ActionPathList(mode='fr'))
    >>> cfg = parser.parse_args(['--list', 'paths.lst')  # Text file with paths
    >>> cfg = parser.parse_args(['--list', '-')          # List from stdin


Comparison operators
====================

It is quite common that when parsing a number, its range should be limited. To
ease these cases the module includes the :class:`.ActionOperators`. Some
examples of arguments that can be added using this action are the following:

.. code-block:: python

    # Larger than zero
    parser.add_argument('--op1', action=yamlargparse.ActionOperators(expr=('>', 0)))
    # Between 0 and 10
    parser.add_argument('--op2', action=yamlargparse.ActionOperators(expr=[('>=', 0), ('<=', 10)]))
    # Either larger than zero or 'off' string
    def int_or_off(x): return x if x == 'off' else int(x)
    parser.add_argument('--op3', action=yamlargparse.ActionOperators(expr=[('>', 0), ('==', 'off')], join='or', type=int_or_off))


Json schemas
============

The :class:`.ActionJsonSchema` class is provided to allow parsing and validation
of values using a json schema. This class requires the `jsonschema
<https://pypi.org/project/jsonschema/>`_ python package. Though note that
jsonschema is not a requirement of yamlargparse, so to use
:class:`.ActionJsonSchema` it is required to explicitly install jsonschema.

Check out the `jsonschema documentation
<https://python-jsonschema.readthedocs.io/>`_ to learn how to write a schema.
The current version of yamlargparse uses Draft7Validator. Parsing an argument
using a json schema is done like in the following example:

.. code-block:: python

    >>> schema = {
    ...     "type" : "object",
    ...     "properties" : {
    ...         "price" : {"type" : "number"},
    ...         "name" : {"type" : "string"},
    ...     },
    ... }

    >>> parser.add_argument('--op', action=yamlargparse.ActionJsonSchema(schema=schema))

    >>> parser.parse_args(['--op', '{"price": 1.5, "name": "cookie"}'])
    namespace(op=namespace(name='cookie', price=1.5))


Yes/No arguments
================

When parsing boolean values from the command line, sometimes it is useful to
define two paired options, one to set to true and the other for false. The
:class:`.ActionYesNo` makes this straightforward. A couple of examples would be:

.. code-block:: python

    # --opt1 for true and --no_opt1 for false.
    parser.add_argument('--op1', action=yamlargparse.ActionYesNo)
    # --with-opt2 for true and --without-opt2 for false.
    parser.add_argument('--with-op2', action=yamlargparse.ActionYesNo(yes_prefix='with-', no_prefix='without-'))


Parsing with another parser
===========================

Sometimes an element in a yaml file could be a path to another yaml file with a
complex structure which should also be parsed. To handle these cases there is
the :class:`.ActionParser` which receives as argument a yamlargparse parser
object. For example:

.. code-block:: python

    parser.add_argument('--complex.node', action=yamlargparse.ActionParser(parser=node_parser))
