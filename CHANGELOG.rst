.. _changelog:

Changelog
=========

All notable changes to this project will be documented in this file. Versions
follow `Semantic Versioning <https://semver.org/>`_
(``<major>.<minor>.<patch>``). Backward incompatible (breaking) changes will
only be introduced in major versions with advance notice in the **Deprecated**
section of releases.


v3.13.0 (2021-??-??)
--------------------

Added
^^^^^
- Inner config file support for subclass type hints in signatures and CLI #57.

Fixed
^^^^^
- With fail_untyped=True use type from default value instead of Any.
- Registered types incorrectly considered subclass types.


v3.12.1 (2021-05-19)
--------------------

Fixed
^^^^^
- :code:`--print_config` can now be given before other arguments without value.
- Fixed conversion of flat namespace to dict when there is a nested empty namespace.
- Fixed issue with get_defaults with default config file and parse_as_dict=False.
- Fixed bug in save which failed when there was an int key.

Changed
^^^^^^^
- :code:`--print_config` now only receives a value with `=` syntax.
- :code:`add_{class,method,function,dataclass}_arguments` now return a list of
  added arguments.


v3.12.0 (2021-05-13)
--------------------

Added
^^^^^
- Path support for fsspec file systems using the 's' mode flag.
- set_config_read_mode function that can enable fsspec for config reading.
- Option for print_config and dump with help as yaml comments.

Changed
^^^^^^^
- print_config only added to parsers when ActionConfigFile is added.

Deprecated
^^^^^^^^^^
- set_url_support functionality now should be done with set_config_read_mode.


v3.11.2 (2021-05-03)
--------------------

Fixed
^^^^^
- Link argument arrow :code:`<=` can be confused as less or equal, changed to
  :code:`<--`.


v3.11.1 (2021-04-30)
--------------------

Fixed
^^^^^
- add_dataclass_arguments not making parameters without default as required #54.
- Removed from signature add methods required option included by mistake.


v3.11.0 (2021-04-27)
--------------------

Added
^^^^^
- CLI now has :code:`--config` options at subcommand and subsubcommand levels.
- CLI now adds subcommands with help string taken from docstrings.
- print_config at subcommand level for global config with implicit subcommands.
- New Path_drw predefined type.
- Type hint arguments now support :code:`nargs='?'`.
- Signature methods can now skip arguments within init_args of subclasses.

Changed
^^^^^^^
- Removed skip_check from ActionPathList which was never implemented.

Deprecated
^^^^^^^^^^
- ActionPath should no longer be used, instead paths are given as type.

Fixed
^^^^^
- Actions not being applied for subsubcommand values.
- handle_subcommands not correctly inferring subsubcommand.


v3.10.1 (2021-04-24)
--------------------

Changed
^^^^^^^
- fail_untyped now adds untyped parameters as type Any and if no default
  then default set to None.

Fixed
^^^^^
- :code:`--*.help` option being added for non-subclass types.
- Iterable and Sequence types not working for python>=3.7 #53.


v3.10.0 (2021-04-19)
--------------------

Added
^^^^^
- set_defaults method now works for arguments within subcommands.
- CLI set_defaults option to allow overriding of defaults.
- CLI return_parser option to ease inclusion in documentation.
- save_path_content attribute to save paths content on config save.
- New `link_arguments` method to derive an argument value from others.
- print_config now includes subclass init_args if class_path given.
- Subclass type hints now also have a :code:`--*.help` option.

Changed
^^^^^^^
- Signature parameters whose name starts with "_" are skipped.
- The repr of Path now has the form `Path_{mode}(`.

Fixed
^^^^^
- CLI now does instantiate_subclasses before running.


v3.9.0 (2021-04-09)
-------------------

Added
^^^^^
- New method add_dataclass_arguments.
- Dataclasses are now supported as a type.
- New predefined type Path_dc.
- Experimental Callable type support.
- Signature methods with nested key can be made required.
- Support for Literal types.
- New option in signatures methods to not fail for untyped required.

Changed
^^^^^^^
- Generation of yaml now uses internally pyyaml's safe_dump.
- New cleaner implementation for type hints support.
- Moved deprecated code to a module specific for this.
- Path types repr now has format Path(rel[, cwd=dir]).
- instantiate_subclasses now always returns a dict.

Deprecated
^^^^^^^^^^
- ActionEnum should no longer be used, instead enums are given as type.

Fixed
^^^^^
- Deserialization of types not being done for nested config files.


v3.8.1 (2021-03-22)
-------------------

Fixed
^^^^^
- Help fails saying required args missing if default config file exists #48.
- ActionYesNo arguments failing when parsing from environment variable #49.


v3.8.0 (2021-03-22)
-------------------

Added
^^^^^
- Path class now supports home prefix '~' #45.
- yaml/json dump kwargs can now be changed via attributes dump_yaml_kwargs and
  dump_json_kwargs.

Changed
^^^^^^^
- Now by default dump/save/print_config preserve the add arguments and argument
  groups order (only CPython>=3.6) #46.
- ActionParser group title now defaults to None if not given #47.
- Add argument with type Enum or type hint giving an action now raises error #45.
- Parser help now also considers default_config_files and shows which config file
  was loaded #47.
- get_default method now also considers default_config_files.
- get_defaults now raises ParserError if default config file not valid.

Fixed
^^^^^
- default_config_files property not removing help group when setting None.


v3.7.0 (2021-03-17)
-------------------

Changed
^^^^^^^
- ActionParser now moves all actions to the parent parser.
- The help of ActionParser arguments is now shown in the main help #41.

Fixed
^^^^^
- Use of required in ActionParser parsers not working #43.
- Nested options with names including dashes not working #42.
- DefaultHelpFormatter not properly using env_prefix to show var names.


v3.6.0 (2021-03-08)
-------------------

Added
^^^^^
- Function to register additional types for use in parsers.
- Type hint support for complex and UUID classes.

Changed
^^^^^^^
- PositiveInt and NonNegativeInt now gives error instead of silently truncating
  when given float.
- Types created with restricted_number_type and restricted_string_type now share
  a common TypeCore base class.

Fixed
^^^^^
- ActionOperators not give error if type already registered.
- List[Tuple] types not working correctly.
- Some nested dicts kept as Namespace by dump.


v3.5.1 (2021-02-26)
-------------------

Fixed
^^^^^
- Parsing of relative paths in default_config_files not working.
- Description of tuple type in the readme.


v3.5.0 (2021-02-12)
-------------------

Added
^^^^^
- Tuples with ellipsis are now supported #40.

Fixed
^^^^^
- Using dict as type incorrectly considered as class requiring class_path.
- Nested tuples were not working correctly #40.


v3.4.1 (2021-02-03)
-------------------

Fixed
^^^^^
- CLI crashed for class method when zero args given after subcommand.
- Options before subcommand provided in config file gave subcommand not given.
- Arguments in groups without help not showing required, type and default.
- Required arguments help incorrectly showed null default value.
- Various improvements and fixes to the readme.


v3.4.0 (2021-02-01)
-------------------

Added
^^^^^
- Save with multifile=True now creates original jsonnet file for ActionJsonnet.
- default_config_files is now a property of parser objects.
- Table in readme to ease understanding of extras requires for optional features #38.

Changed
^^^^^^^
- Save with multifile=True uses file extension to choose json or yaml format.

Fixed
^^^^^
- Better exception message when using ActionJsonSchema and jsonschema not installed #38.


v3.3.2 (2021-01-22)
-------------------

Fixed
^^^^^
- Changed actions so that keyword arguments are visible in API.
- Fixed save method short description which was copy paste of dump.
- Added missing docstring in instantiate_subclasses method.
- Fixed crash when using :code:`--help` and ActionConfigFile not given help string.
- Standardized capitalization and punctuation of: help, config, version.


v3.3.1 (2021-01-08)
-------------------

Fixed
^^^^^
- instantiate_subclasses work properly when init_args not present.
- Addressed a couple of issues pointed out by sonarcloud.


v3.3.0 (2021-01-08)
-------------------

Added
^^^^^
- New add_subclass_arguments method to add as type with a specific help option.


v3.2.1 (2020-12-30)
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
- Avoid duplicate config check failure message with subcommands.


v3.1.0 (2020-12-09)
-------------------

Added
^^^^^
- Support for multiple levels of subcommands #29.
- Default description of subcommands explaining use of :code:`--help`.


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
- Parsers now include :code:`--print_config` option to dump defaults.
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
