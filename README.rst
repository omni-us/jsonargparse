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
--------

- Parsers are configured just like with python's argparse, thus it has a gentile learning curve.

- Not exclusively intended for parsing command line arguments. The module has functions to parse environment variables and yaml config files.

- Configuration settings are overridden based on the following precedence.

  - **Parsing command line:** command line arguments (might include config file) > environment variables > defaults.
  - **Parsing yaml:** config file values > environment variables > defaults.
  - **Parsing environment:** environment variables > defaults.

- Support of nested namespaces to make it possible to parse yaml with non-flat hierarchies.


Contact
-------

- Mauricio Villegas <mauricio@omnius.com>
