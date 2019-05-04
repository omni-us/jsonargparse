yamlargparse python module
==========================

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

- Not exclusively intended for parsing command line arguments. The module has functions to parse environment variables and yaml config files.

- Support for nested namespaces which makes it possible to parse yaml with non-flat hierarchies.

- Configuration settings are overridden based on the following precedence.

  - **Parsing command line:** command line arguments (might include config file) > environment variables > defaults.
  - **Parsing yaml:** config file values > environment variables > defaults.
  - **Parsing environment:** environment variables > defaults.


Using the module
================

A parser is created just like it is done with argparse. You import the module,
create a parser object and then add arguments to it. A simple example would be:

.. code-block:: python

    import yamlargparse
    parser = yamlargparse.ArgumentParser(
        prog = 'app',
        description = 'Description for my app.')

    parser.add_argument('--opt1',
        type = int,
        default = 0,
        help = 'Help for option 1.')

    parser.add_argument('--opt2',
        type = float,
        default = 1.0,
        help = 'Help for option 2.')


After creating the parser, you can use it to parse command line arguments with
the :code:`parse_args` function, after which you get an object with the parsed
values or defaults available as attributes. For illustrative purposes giving to
:code:`parse_args` a list of arguments (instead of automatically getting them
from the command line arguments), with the parser from above you would observe:

.. code-block:: python

    >>> cfg = parser.parse_args(['--opt2', '2.3'])
    >>> cfg.opt1, type(cfg.opt1)
    (0, <class 'int'>)
    >>> cfg.opt2, type(cfg.opt2)
    (2.3, <class 'float'>)


.. _nested-namespaces-label:

Nested namespaces
=================

A difference with respect to the basic argparse is that it by using dot notation
in the argument names, you can define a hierarchy of nested namespaces. So for
example you could do the following:

.. code-block:: python

    >>> parser = yamlargparse.ArgumentParser(prog='app')
    >>> parser.add_argument('--lev1.opt1', type=str, default='from default 1')
    >>> parser.add_argument('--lev1.opt2', type=str, default='from default 2')
    >>> cfg = parser.get_defaults()
    >>> cfg.lev1.opt1
    'from default 2'
    >>> cfg.lev1.opt2
    'from default 2'


Environment variables
=====================

The yamlargparse parsers by default also get values from environment variables.
In the case of environment variables, the parser checks existing variables whose
name is of the form :code:`[PROG_][LEV__]*OPT`, that is all in upper case, first
only if set the name of the program followed by underscore and then the argument
name replacing dots with two underscores. Using the parser from the
:ref:`nested-namespaces-label` section above, in your shell you would set the
environment variables as:

.. code-block:: bash

    export APP_LEV1__OPT1='from env 1'
    export APP_LEV1__OPT2='from env 2'

Then in python the parser would use these variables, unless overridden by the
command line arguments, that is:

.. code-block:: python

    >>> parser = yamlargparse.ArgumentParser(prog='app')
    >>> parser.add_argument('--lev1.opt1', type=str, default='from default 1')
    >>> parser.add_argument('--lev1.opt2', type=str, default='from default 2')
    >>> cfg = parser.parse_args(['--lev1.opt1', 'from arg 1'])
    >>> cfg.lev1.opt1
    'from arg 1'
    >>> cfg.lev1.opt2
    'from env 2'

There is also the :code:`parse_env` function to only parse environment
variables, which might be useful for some use cases in which there is no command
line call involved.


YAML configuration files
========================

An important feature of this module is the parsing of yaml files. The dot
notation hierarchy of the arguments (see :ref:`nested-namespaces-label`) are
used for the expected structure of the yaml files.

When parsing command line arguments, it is possible to add a yaml configuration
file path argument. The yaml file would be read and parsed in the specific
position among the command line arguments, so the arguments after would override
the values from the yaml file. Again using the parser from the
:ref:`nested-namespaces-label` section above, for example we could have the
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
    >>> parser.add_argument('--lev1.opt1', type=str, default='from default 1')
    >>> parser.add_argument('--lev1.opt2', type=str, default='from default 2')
    >>> parser.add_argument('--cfg', action=yamlargparse.ActionConfigFile)
    >>> cfg = parser.parse_args(['--lev1.opt1', 'from arg 1', '--cfg', 'example.yaml', '--lev1.opt2', 'from arg 2'])
    >>> cfg.lev1.opt1
    'from yaml 1'
    >>> cfg.lev1.opt2
    'from arg 2'

There are also functions :code:`parse_yaml` and :code:`parse_yaml_from_string`
to only parse a yaml file or yaml contained in a string.
