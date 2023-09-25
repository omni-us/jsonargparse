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

Docs: https://jsonargparse.readthedocs.io/ | Source: https://github.com/omni-us/jsonargparse/

``jsonargparse`` is a library for creating command-line interfaces (CLIs) and
making Python apps easily configurable. It is a well-maintained project with
frequent releases, adhering to high standards of development: semantic
versioning, deprecation periods, changelog, automated testing, and full test
coverage.

Although ``jsonargparse`` might not be widely recognized yet, it already boasts
a `substantial user base
<https://github.com/omni-us/jsonargparse/network/dependents>`__. Most notably,
it serves as the framework behind pytorch-lightning's `LightningCLI
<https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html>`__.


Features
--------

``jsonargparse`` is user-friendly and encourages the development of **clean,
high-quality code**. It encompasses numerous powerful features, some unique to
``jsonargparse``, while also combining advantages found in similar packages:

- **Automatic** creation of CLIs, like `Fire
  <https://pypi.org/project/fire/>`__, `Typer
  <https://pypi.org/project/typer/>`__, `Clize
  <https://pypi.org/project/clize/>`__ and `Tyro
  <https://pypi.org/project/tyro/>`__.

- Use **type hints** for argument validation, like `Typer
  <https://pypi.org/project/typer/>`__, `Tap
  <https://pypi.org/project/typed-argument-parser/>`__ and `Tyro
  <https://pypi.org/project/tyro/>`__.

- Use of **docstrings** for automatic generation of help, like `Tap
  <https://pypi.org/project/typed-argument-parser/>`__, `Tyro
  <https://pypi.org/project/tyro/>`__ and `SimpleParsing
  <https://pypi.org/project/simple-parsing/>`__.

- Parse from **configuration files** and **environment variables**, like
  `OmegaConf <https://pypi.org/project/omegaconf/>`__, `dynaconf
  <https://pypi.org/project/dynaconf/>`__, `confuse
  <https://pypi.org/project/confuse/>`__ and `configargparse
  <https://pypi.org/project/ConfigArgParse/>`__.

- **Dataclasses** support, like `SimpleParsing
  <https://pypi.org/project/simple-parsing/>`__ and `Tyro
  <https://pypi.org/project/tyro/>`__.

Other notable features include:

- **Extensive type hint support:** nested types (union, optional), containers
  (list, dict, etc.), user-defined generics, restricted types (regex, numbers),
  paths, URLs, types from stubs (``*.pyi``), future annotations (PEP `563
  <https://peps.python.org/pep-0563/>`__), and backports (PEPs `604
  <https://peps.python.org/pep-0604>`__/`585
  <https://peps.python.org/pep-0585>`__).

- **Keyword arguments introspection:** resolving of parameters used via
  ``**kwargs``.

- **Dependency injection:** support types that expect a class instance and
  callables that return a class instance.

- **Structured configs:** parse config files with more understandable non-flat
  hierarchies.

- **Config file formats:** `json <https://www.json.org/>`__, `yaml
  <https://yaml.org/>`__, `jsonnet <https://jsonnet.org/>`__ and extendible to
  more formats.

- **Relative paths:** within config files and parsing of config paths referenced
  inside other configs.

- **Argument linking:** directing parsed values to multiple parameters,
  preventing unnecessary interpolation in configs.


Design principles
-----------------

- **Non-intrusive/decoupled:**

  There is no requirement for unrelated modifications throughout a codebase,
  maintaining the `separation of concerns principle
  <https://en.wikipedia.org/wiki/Separation_of_concerns>`__. In simpler terms,
  changes should make sense even without the CLI. No need to inherit from a
  special class, add decorators, or use CLI-specific type hints.

- **Minimal boilerplate:**

  A recommended practice is to write code with function/class parameters having
  meaningful names, accurate type hints, and descriptive docstrings. Reuse these
  wherever they appear to automatically generate the CLI, following the `don't
  repeat yourself principle
  <https://en.wikipedia.org/wiki/Don%27t_repeat_yourself>`__. A notable
  advantage is that when parameters are added or types changed, the CLI will
  remain synchronized, avoiding the need to update the CLI's implementation.

- **Dependency injection:**

  Using as type hint a class or a callable that instantiates a class, a practice
  known as `dependency injection
  <https://en.wikipedia.org/wiki/Dependency_injection>`__, is a sound design
  pattern for developing loosely coupled and highly configurable software. Such
  type hints should be supported with minimal restrictions.


.. _installation:

Installation
============

You can install using `pip <https://pypi.org/project/jsonargparse/>`__ as:

.. code-block:: bash

    pip install jsonargparse

By default the only dependency that jsonargparse installs is `PyYAML
<https://pypi.org/project/PyYAML/>`__. However, several optional features can be
enabled by specifying any of the following extras requires: ``signatures``,
``jsonschema``, ``jsonnet``, ``urls``, ``fsspec``, ``ruyaml``, ``omegaconf`` and
``argcomplete``. There is also the ``all`` extras require to enable all optional
features. Installing jsonargparse with extras require is as follows:

.. code-block:: bash

    pip install "jsonargparse[signatures,urls]"  # Enable signatures and URLs features
    pip install "jsonargparse[all]"              # Enable all optional features
