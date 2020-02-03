.. image:: https://circleci.com/gh/omni-us/jsonargparse.svg?style=svg
    :target: https://circleci.com/gh/omni-us/jsonargparse
.. image:: https://codecov.io/gh/omni-us/jsonargparse/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/omni-us/jsonargparse
.. image:: https://badge.fury.io/py/jsonargparse.svg
    :target: https://badge.fury.io/py/jsonargparse
.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg
    :target: https://github.com/omni-us/jsonargparse


jsonargparse (former yamlargparse)
==================================

https://omni-us.github.io/jsonargparse/

This module is an extension to python's argparse which simplifies parsing of
configuration options from command line arguments, json supersets (`yaml
<https://yaml.org/>`__ or `jsonnet <https://jsonnet.org/>`__) configuration
files, environment variables and hard-coded defaults.

The aim is similar to other projects such as `configargparse
<https://pypi.org/project/ConfigArgParse/>`__, `yconf
<https://pypi.org/project/yconf/>`__ and `confuse
<https://pypi.org/project/confuse/>`__. The obvious question is, why yet another
module similar to many already existing ones? The answer is simply that none of
the existing projects had the exact features we wanted and after analyzing the
alternatives it seemed simpler to create a new module.


Features
========

- Parsers are configured just like with python's argparse, thus it has a gentle
  learning curve.

- Not exclusively intended for parsing command line arguments. The main focus is
  parsing yaml or jsonnet configuration files and not necessarily from a command
  line tool.

- Support for nested namespaces which makes it possible to parse config files
  with non-flat hierarchies.

- Support for two popular supersets of json, making config files more versatile
  and powerful.

- Parsing of relative paths within config files and path lists.

- Several convenient action classes to ease common parsing use cases (paths,
  comparison operators, json schemas, ...).

- Default behavior is not identical to argparse, though it is possible to
  configure it to be identical. The main differences are:

  - When parsing fails :class:`ParserError` is raised, instead of printing usage
    and program exit.
  - To modify the behavior for parsing errors (e.g. print usage) an error
    handler function can be provided.

- Configuration values are overridden based on the following precedence.

  - **Parsing command line:** command line arguments (might include config file)
    > environment variables > default config file > defaults.
  - **Parsing files:** config file > environment variables > default config file
    > defaults.
  - **Parsing environment:** environment variables > default config file >
    defaults.


Basic usage
===========

A parser is created just like it is done with argparse. You import the module,
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
the :func:`jsonargparse.ArgumentParser.parse_args` function, after which you get
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

If the parsing fails a :class:`ParserError` is raised, so depending on the use
case it might be necessary to catch it.

.. code-block:: python

    >>> try:
    ...     cfg = parser.parse_args(['--opt2', 'four'])
    ... except jsonargparse.ParserError as ex:
    ...     print('parser error: '+str(ex))
    ...
    parser error: argument --opt2: invalid float value: 'four'

To get the default behavior of argparse the ArgumentParser can be initialized as
follows:

.. code-block:: python

    parser = ArgumentParser(
        prog='app',
        error_handler='usage_and_exit_error_handler',
        description='Description for my app.')


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


Environment variables
=====================

The jsonargparse parsers can also get values from environment variables. The
parser checks existing environment variables whose name is of the form
:code:`[PREFIX_][LEV__]*OPT`, that is all in upper case, first a prefix (set by
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

Note that when creating the parser, :code:`default_env=True` was given as
argument. By default :func:`jsonargparse.ArgumentParser.parse_args` does not
check environment variables, so it has to be enabled explicitly.

There is also the :func:`jsonargparse.ArgumentParser.parse_env` function to only
parse environment variables, which might be useful for some use cases in which
there is no command line call involved.

If a parser includes an :class:`.ActionConfigFile` argument, then the
environment variable for this config file will be checked before all the other
environment variables.


Configuration files
===================

An important feature of this module is the parsing of yaml/json files. The dot
notation hierarchy of the arguments (see :ref:`nested-namespaces`) are used for
the expected structure in the config files.

When creating the :class:`.ArgumentParser` the :code:`default_config_files`
argument can be given to specify patterns to search for configuration files.
Only the first matched config file is parsed.

When parsing command line arguments, it is possible to add a configuration file
path argument. The config file would be read and parsed in the specific position
among the command line arguments, so the arguments after would override the
values from the configuration file. If the config argument can be given multiple
times, each overriding the values of the previous. Again using the parser from
the :ref:`nested-namespaces` section above, for example we could have the
following config file in yaml format:

.. code-block:: yaml

    # File: example.yaml
    lev1:
      opt1: from yaml 1
      opt2: from yaml 2

Then in python adding a yaml file argument and parsing some example arguments,
the following would be observed:

.. code-block:: python

    >>> from jsonargparse import ArgumentParser, ActionConfigFile
    >>> parser = ArgumentParser()
    >>> parser.add_argument('--lev1.opt1', default='from default 1')
    >>> parser.add_argument('--lev1.opt2', default='from default 2')
    >>> parser.add_argument('--cfg', action=ActionConfigFile)
    >>> cfg = parser.parse_args(['--lev1.opt1', 'from arg 1', '--cfg', 'example.yaml', '--lev1.opt2', 'from arg 2'])
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

The config file could also be provided as an environment variable as explained
in section :ref:`environment-variables`. The configuration file environment
variable is the first one to be parsed. So any other argument provided through
environment variables would override the config file one.

A configuration file or string can also be parsed without parsing command line
arguments. The functions for this are
:func:`jsonargparse.ArgumentParser.parse_path` and
:func:`jsonargparse.ArgumentParser.parse_string` to parse a config file or a
config contained in a string respectively.


Json schemas
============

The :class:`.ActionJsonSchema` class is provided to allow parsing and validation
of values using a json schema. This class requires the `jsonschema
<https://pypi.org/project/jsonschema/>`__ python package. Though note that
jsonschema is not a requirement of the minimal jsonargparse install. To enable
this functionality install the module with the *all* extras requires as:

.. code-block:: bash

    $ pip3 install jsonargparse[all]

Check out the `jsonschema documentation
<https://python-jsonschema.readthedocs.io/>`__ to learn how to write a schema.
The current version of jsonargparse uses Draft4Validator. Parsing an argument
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
    namespace(op=namespace(name='cookie', price=1.5))

Instead of giving a json string as argument value, it is also possible to
provide a path to a json/yaml file, which would be loaded and validated against
the schema. If the schema defines default values, these will be used by the
parser to initialize the config values that are not specified. When adding an
argument with the :class:`.ActionJsonSchema` action, you can use "%s" in the
:code:`help` string so that in that position the schema will be printed.


Jsonnet files
=============

The Jsonnet support requires `jsonschema
<https://pypi.org/project/jsonschema/>`__ and `jsonnet
<https://pypi.org/project/jsonnet/>`__ python packages which are not included
with minimal jsonargparse install. To enable this functionality install
jsonargparse with the *all* extras requires as:

.. code-block:: bash

    $ pip3 install jsonargparse[all]

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
argument which would be a path to a jsonnet file. Moreover, another argument can
be specified as the source for any external variables required, which would be
either a path to or a string containing a json dictionary of variables. Its use
would be as follows:

.. code-block:: python

    from jsonargparse import ArgumentParser, ActionJsonnet, ActionJsonnetExtVars
    parser = ArgumentParser()
    parser.add_argument('--in_ext_vars',
        action=ActionJsonnetExtVars)
    parser.add_argument('--in_jsonnet',
        action=ActionJsonnet(ext_vars='in_ext_vars'))

For example, if a jsonnet file required some external variable :code:`param`,
then the jsonnet and the external variable could be given as:

.. code-block:: python

        cfg = parser.parse_args(['--in_ext_vars', '{"param": 123}', '--in_jsonnet', 'path_to_jsonnet'])

Note that the external variables argument must be provided before the jsonnet
path so that this dictionary already exists when parsing the jsonnet.

The :class:`.ActionJsonnet` class also accepts as argument a json schema, in
which case the jsonnet would be validated against this schema right after
parsing.


Parsing paths
=============

For some use cases it is necessary to parse file paths, checking its existence
and access permissions, but not necessarily opening the file. Moreover, a file
path could be included in a config file as relative with respect to the config
file's location. After parsing it should be easy to access the parsed file path
without having to consider the location of the config file. To help in these
situations jsonargparse includes the :class:`.ActionPath` and the
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

    >>> from jsonargparse import ArgumentParser, ActionPath
    >>> parser = ArgumentParser()
    >>> parser.add_argument('--databases.info', action=ActionPath(mode='fr'))
    >>> cfg = parser.parse_path('app/config.yaml')

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

    >>> from jsonargparse import ActionPathList
    >>> parser.add_argument('--list', action=ActionPathList(mode='fr'))
    >>> cfg = parser.parse_args(['--list', 'paths.lst')  # Text file with paths
    >>> cfg = parser.parse_args(['--list', '-')          # List from stdin

If :code:`nargs='+'` is given to :code:`add_argument` then a single list is
generated including all paths in all lists provided.


Comparison operators
====================

It is quite common that when parsing a number, its range should be limited. To
ease these cases the module includes the :class:`.ActionOperators`. Some
examples of arguments that can be added using this action are the following:

.. code-block:: python

    from jsonargparse import ActionOperators
    # Larger than zero
    parser.add_argument('--op1', action=ActionOperators(expr=('>', 0)))
    # Between 0 and 10
    parser.add_argument('--op2', action=ActionOperators(expr=[('>=', 0), ('<=', 10)]))
    # Either larger than zero or 'off' string
    def int_or_off(x): return x if x == 'off' else int(x)
    parser.add_argument('--op3', action=ActionOperators(expr=[('>', 0), ('==', 'off')], join='or', type=int_or_off))


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
    namespace(op1=True, op2=False)

Sometimes it is also useful to define two paired options, one to set
:code:`True` and the other to set :code:`False`. The :class:`.ActionYesNo` class
makes this straightforward. A couple of examples would be:

.. code-block:: python

    from jsonargparse import ActionYesNo
    # --opt1 for true and --no_opt1 for false.
    parser.add_argument('--op1', action=ActionYesNo)
    # --with-opt2 for true and --without-opt2 for false.
    parser.add_argument('--with-op2', action=ActionYesNo(yes_prefix='with-', no_prefix='without-'))

If the :class:`.ActionYesNo` class is used in conjunction with
:code:`nargs='?'` the options can also be set by giving as value any of
:code:`{'true', 'yes', 'false', 'no'}`.


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
        action=ActionParser(parser=inner_parser))

When using the :class:`.ActionParser` class, the value of the node in a config
file can be either the complex node itself, or the path to a file which will be
loaded and parsed with the corresponding inner parser. Naturally using
:class:`.ActionConfigFile` to parse a complete config file will parse the inner
nodes correctly.

From the command line the help of the inner parsers can be shown by calling the
tool with a prefixed help command, that is, for the example above it would be
:code:`--inner.node.help`.

Regarding environment variables, the prefix of the outer parser will be used to
populate the leaf nodes of the inner parser. In the example above, if
:code:`inner_parser` is used to parse environment variables, then as normal
:code:`APP1_OP1` would be checked to populate option :code:`op1`. But if
:code:`outer_parser` is used, then :code:`APP2_INNER__NODE__OP1` would be
checked to populate :code:`inner.node.op1`.

An important detail to note is that the parsers that are given to
:class:`.ActionParser` are internally modified. So they should be instantiated
exclusively for the :class:`.ActionParser` and not used standalone.


Contributing
============

Contributions to the jsonargparse package are very welcome, be it just to create
`issues <https://github.com/omni-us/jsonargparse/issues>`_ for reporting bugs
and proposing enhancements, or more directly by creating `pull requests
<https://github.com/omni-us/jsonargparse/pulls>`_.

If you intend to work with the source code, note that this project does not
include any :code:`requirements.txt` file. This is by intention. To make it very
clear what are the requirements for different use cases, all the requirements of
the project are stored in the file :code:`setup.cfg`. The basic runtime
requirements are defined in section :code:`[options]` in the
:code:`install_requires` entry. All optional requirements are stored in section
:code:`[options.extras_require]` in the :code:`all` entry. And finally there are
:code:`dev` and :code:`doc` entries in the same :code:`[options.extras_require]`
section which lists requirements for development and documentation building.

The recommended way to work with the source code is the following. First clone
the repository, then create a virtual environment, activate it and finally
install the development requirements. More precisely the steps would be:

.. code-block:: bash

    git clone https://github.com/omni-us/jsonargparse.git
    cd jsonargparse
    virtualenv -p python3 venv
    . venv/bin/activate

The crucial step is installing the requirements which would be done by running:

.. code-block:: bash

    pip install --editable .[dev,doc,all]
