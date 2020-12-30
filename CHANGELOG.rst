.. _changelog:

Changelog
=========

All notable changes to this project will be documented in this file. Versions
follow `Semantic Versioning <https://semver.org/>`_
(``<major>.<minor>.<patch>``). Backward incompatible (breaking) changes will
only be introduced in major versions with advance notice in the **Deprecated**
section of releases.

v3.X.X (YYYY-MM-DD)
-------------------

Added
^^^^^
- Automatic Optional for arguments with default None #30.
- CLI now supports running methods from classes.
- Signature arguments can now be loaded from independent config files #32.
- add_argument now supports enable_path for type based on jsonschema.
- print_config can now be given as value skip_null to exclude null entries.

Changed
^^^^^^^
- Improved description of parser used as standalone and for ActionParser #34.
- Removed :code:`__cwd__` and top level :code:`__path__` that were not needed.

Fixed
^^^^^
- ActionYesNo argument in help the type is now bool.
- Correctly skip self in add_method_arguments for inherited methods.
- Prevent failure of dump in cleanup_actions due to new _ActionConfigLoad.
- Prevent failure in save_paths for dict with int keys.


v3.1.0 (2020-12-09)
-------------------

Added
^^^^^
- Support for multiple levels of subcommands #29.
- Default description of subcommands explaining use of --help.


v3.0.1 (2020-12-02)
-------------------

Fixed
^^^^^
- add_class_arguments incorrectly added arguments from :code:`__call__` instead
  of :code:`__init__` for callable classes.


v3.0.0 (2020-12-01)
-------------------

Added
^^^^^
- Functions to add arguments from classes, methods and functions.
- CLI function that allows creating a line command line interface with a single
  line of code inspired by Fire.
- Typing module that includes predefined types and type generator functions
  for paths and restricted numbers/strings.
- Extended support to add_argument type to allow complex type hints.
- Parsers now include --print_config option to dump defaults.
- Support argcomplete for tab completion of arguments.

Changed
^^^^^^^
- ArgumentParsers by default now use as error_handler the
  usage_and_exit_error_handler.
- error_handler and formatter_class no longer accept as value a string.
- Changed SimpleNamespace to Namespace to avoid unnecessary differences with
  argparse.

Deprecated
^^^^^^^^^^
- ActionOperators should no longer be used, the new alternative is
  restricted number types.


v2.X.X
------

The change log was introduced in v3.0.0. For details of the changes for previous
versions take a look at the git log. It more or less reads like a change log.
