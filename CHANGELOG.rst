Changelog
=========

All notable changes to this project will be documented in this file. Versions
follow `Semantic Versioning <https://semver.org/>`_
(``<major>.<minor>.<patch>``). Backward incompatible (breaking) changes will
only be introduced in major versions with advance notice in the **Deprecated**
section of releases.

The semantic versioning only considers the public API as described in
:ref:`api-ref`. Components not mentioned in :ref:`api-ref` or different import
paths are considered internals and can change in minor and patch releases.


v4.25.0 (2023-09-??)
--------------------

Changed
^^^^^^^
- Provide a more informative error message to remind user to select
  and provide a subcommand when a subcommand is required but not
  provided (`#371<https://github.com/omni-us/jsonargparse/pull/371>`__).
- Removed support for python 3.6.


v4.24.1 (2023-09-06)
--------------------

Fixed
^^^^^
- Remove private ``linked_targets`` parameter from API Reference (`#317
  <https://github.com/omni-us/jsonargparse/issues/317>`__).
- Dataclass nested in list not setting defaults (`#357
  <https://github.com/omni-us/jsonargparse/issues/357>`__)
- AST resolver ``kwargs.pop()`` with conflicting defaults not setting the
  conditional default (`#362
  <https://github.com/omni-us/jsonargparse/issues/362>`__).
- ``ActionJsonSchema`` not setting correctly defaults when schema uses
  ``oneOf``.
- Recommended ``print_config`` steps not working when ``default_config_files``
  used due to the config file initially being empty (`#367
  <https://github.com/omni-us/jsonargparse/issues/367>`__).


v4.24.0 (2023-08-23)
--------------------

Added
^^^^^
- New option in ``dump`` for including link targets.
- Support ``decimal.Decimal`` as a type.
- ``CLI`` now accepts components as a dict, such that the keys define names of
  the subcommands (`#334
  <https://github.com/omni-us/jsonargparse/issues/334>`__).
- Resolve types that use ``TYPE_CHECKING`` blocks (`#337 comment
  <https://github.com/omni-us/jsonargparse/issues/337#issuecomment-1665055459>`__).
- Improved resolving of nested forward references in types.
- The ``ext_vars`` for an ``ActionJsonnet`` argument can now have a default.
- New method ``ArgumentParser.add_instantiator`` that enables developers to
  implement custom instantiation (`#170
  <https://github.com/omni-us/jsonargparse/issues/170>`__).

Deprecated
^^^^^^^^^^
- ``ActionJsonnetExtVars`` is deprecated and will be removed in v5.0.0. Instead
  use ``type=dict``.


v4.23.1 (2023-08-04)
--------------------

Fixed
^^^^^
- ``save`` fails when a link target is a required parameter nested in a subclass
  (`#332 <https://github.com/omni-us/jsonargparse/issues/332>`__).
- ``typing.Literal`` types skipped when typing_extensions is installed
  (`lightning#18184 <https://github.com/Lightning-AI/lightning/pull/18184>`__).
- ``class_from_function`` failing when called on the same function multiple
  times (`lightning#18180
  <https://github.com/Lightning-AI/lightning/issues/18180>`__).
- Prevent showing errors when running ``ps`` on windows.


v4.23.0 (2023-07-27)
--------------------

Added
^^^^^
- Classes created with ``class_from_function`` now have a valid import path
  (`#309 <https://github.com/omni-us/jsonargparse/issues/309>`__).

Fixed
^^^^^
- Invalid environment variable names when ``env_prefix`` is derived from
  a ``prog`` containing dashes.
- Pylance unable to resolve types from ``jsonargparse.typing``.
- Inconsistent ``ARG:`` and missing ``ENV:`` in help when ``default_env=True``.
- ``typing.Literal`` types skipped on python 3.9 when typing_extensions is
  installed (`lightning#18125 comment
  <https://github.com/Lightning-AI/lightning/pull/18125#issuecomment-1644797707>`__).

Changed
^^^^^^^
- Subcommands main parser help changes:
    - Set notation of subcommands choices now only included in usage.
    - In subcommands section, now each subcommand is always shown separately,
      including the name, and if available aliases and help.
    - When ``default_env=True`` include subcommand environment variable name.


v4.22.1 (2023-07-07)
--------------------

Fixed
^^^^^
- Parameter without default and type optional incorrectly added as a required
  argument (`#312 <https://github.com/omni-us/jsonargparse/issues/312>`__).
- ``class_from_function`` not failing when return annotation is missing.
- ``add_subclass_arguments`` with single base class and no docstring,
  incorrectly shown as tuple in help.
- When all arguments of a group are derived from links, a config load option is
  incorrectly shown in help.
- Printing help fails for parsers that have a link whose target is an argument
  lacking type and help.


v4.22.0 (2023-06-23)
--------------------

Added
^^^^^
- Parameters that receive a path now also accept ``os.PathLike`` type.
- ``class_from_function`` now supports ``func_return`` parameter to specify the
  return type of the function (`lightning-flash#1564 comment
  <https://github.com/Lightning-Universe/lightning-flash/pull/1564#discussion_r1218147330>`__).
- Support for postponed evaluation of annotations PEP `563
  <https://peps.python.org/pep-0563/>`__ ``from __future__ import annotations``
  (`#120 <https://github.com/omni-us/jsonargparse/issues/120>`__).
- Backport types in python<=3.9 to support PEP `585
  <https://peps.python.org/pep-0585/>`__ and `604
  <https://peps.python.org/pep-0604/>`__ for postponed evaluation of annotations
  (`#120 <https://github.com/omni-us/jsonargparse/issues/120>`__).
- Support for ``range`` as a type.

Fixed
^^^^^
- Regular expressions vulnerable to polynomial runtime due to backtracking.
- ``attrs`` fields with factory default causes parse to fail (`#299
  <https://github.com/omni-us/jsonargparse/issues/299>`__).
- Stop subclass dive if you hit bad import (`#304
  <https://github.com/omni-us/jsonargparse/issues/304>`__).

Changed
^^^^^^^
- Added ``_`` prefix to module names to be explicit about non-public API.

Deprecated
^^^^^^^^^^
- Importing from original non-public module paths (without ``_`` prefix) now
  gives a ``DeprecationWarning``. From v5.0.0 these imports will fail.


v4.21.2 (2023-06-08)
--------------------

Fixed
^^^^^
- Failure for nested argument in optional dataclass type (`#289
  <https://github.com/omni-us/jsonargparse/issues/289>`__).
- Argument links applied on parse silently ignored when the source validation
  fails.


v4.21.1 (2023-05-09)
--------------------

Fixed
^^^^^
- AST resolver not working for dict used in a method when the dict is created
  using the curly braces syntax.
- Failure on multiple deep arguments linked on instantiation (`#275
  <https://github.com/omni-us/jsonargparse/issues/275>`__).


v4.21.0 (2023-04-21)
--------------------

Added
^^^^^
- Support for dataclasses nested in a type (`#243
  <https://github.com/omni-us/jsonargparse/issues/243>`__).
- Support for pydantic `models <https://docs.pydantic.dev/usage/models/>`__ and
  attrs `define <https://www.attrs.org/en/stable/examples.html>`__ similar to
  dataclasses.
- Support for `pydantic types
  <https://docs.pydantic.dev/usage/types/#pydantic-types>`__.
- Backport type stubs in python<=3.9 to support PEP `585
  <https://peps.python.org/pep-0585/>`__ and `604
  <https://peps.python.org/pep-0604/>`__ syntax.

Fixed
^^^^^
- `str` parameter in subclass incorrectly parsed as dict with implicit `null`
  value (`#262 <https://github.com/omni-us/jsonargparse/issues/262>`__).
- Wrong error indentation for subclass in union (`lightning#17254
  <https://github.com/Lightning-AI/lightning/issues/17254>`__).
- ``dataclass`` from pydantic not working (`#100 comment
  <https://github.com/omni-us/jsonargparse/issues/100#issuecomment-1408413796>`__).
- ``add_dataclass_arguments`` not forwarding ``sub_configs`` parameter.
- Failure to instantiate nested class group without arguments (`lightning#17263
  <https://github.com/Lightning-AI/lightning/issues/17263>`__).

Changed
^^^^^^^
- Switched from ``setup.cfg`` to ``pyproject.toml`` for configuration.
- Removed ``build_sphinx`` from ``setup.py`` and documented how to build.
- Include enum members in error when invalid value is given
  (`lightning#17247
  <https://github.com/Lightning-AI/lightning/issues/17247>`__).
- The ``signatures`` extras now installs the ``typing-extensions`` package on
  python<=3.9.
- ``CLI`` now when given a class without methods, the class instance is
  returned.

Deprecated
^^^^^^^^^^
- Support for python 3.6 will be removed in v5.0.0. New features added in
  >=4.21.0 releases are not guaranteed to work in python 3.6.


v4.20.1 (2023-03-30)
--------------------

Fixed
^^^^^
- Dump not working for partial callable with return instance
  (`lightning#15340 comment
  <https://github.com/Lightning-AI/lightning/issues/15340#issuecomment-1439203008>`__).
- Allow ``discard_init_args_on_class_path_change`` to handle more nested
  contexts (`#247 <https://github.com/omni-us/jsonargparse/issues/247>`__).
- Failure with dataclasses that have field with ``init=False`` (`#252
  <https://github.com/omni-us/jsonargparse/issues/252>`__).
- Failure when setting individual dict key values for subclasses and
  ``.init_args.`` is included in argument (`#251
  <https://github.com/omni-us/jsonargparse/issues/251>`__).


v4.20.0 (2023-02-20)
--------------------

Added
^^^^^
- ``CLI`` support for callable class instances (`#238
  <https://github.com/omni-us/jsonargparse/issues/238>`__).
- ``add_dataclass_arguments`` now supports the ``fail_untyped`` parameter (`#241
  <https://github.com/omni-us/jsonargparse/issues/241>`__).

Fixed
^^^^^
- ``add_subcommands`` fails when parser has required argument and default config
  available (`#232 <https://github.com/omni-us/jsonargparse/issues/232>`__).

Changed
^^^^^^^
- When parsing fails, now ``argparse.ArgumentError`` is raised instead of
  ``ParserError``.
- Improved error messages when ``fail_untyped=True`` (`#137
  <https://github.com/omni-us/jsonargparse/issues/137>`__).
- ``CLI`` no longer uses the module's docstring as main parser description (`#245
  <https://github.com/omni-us/jsonargparse/issues/245>`__).

Deprecated
^^^^^^^^^^
- Path ``skip_check`` parameter is deprecated and will be removed in v5.0.0.
  Instead use as type ``str`` or ``os.PathLike``.
- Modifying Path attributes is deprecated. In v5.0.0 they will be properties
  without a setter and two renamed: ``rel_path -> relative`` and ``abs_path ->
  absolute``.
- ``ActionPathList`` is deprecated and will be removed in v5.0.0. Instead use as
  type ``List[<path_type>]`` with ``enable_path=True``.
- ``ArgumentParser.error_handler`` is deprecated and will be removed in v5.0.0.
  Instead use the new exit_on_error parameter from argparse.


v4.19.0 (2022-12-27)
--------------------

Added
^^^^^
- ``CLI`` now supports the ``fail_untyped`` and ``parser_class`` parameters.
- ``bytes`` and ``bytearray`` registered on first use and decodes from standard
  Base64.
- Support getting the import path of variables in modules, e.g.
  ``random.randint``.
- Specific error messages for when an argument link uses as source the target of
  a previous parse link and vice versa (`#208
  <https://github.com/omni-us/jsonargparse/issues/208>`__).
- New resolver that identifies parameter types from stub files ``*.pyi``.
- Support for relative paths within remote fsspec/url config files.
- New context manager methods for path types: ``open`` and
  ``relative_path_context``.
- Path types now implement the ``os.PathLike`` protocol.
- New path mode ``cc`` to not require the parent directory to exists but that it
  can be created.
- The parent parser class is now used to create internal parsers (`#171
  <https://github.com/omni-us/jsonargparse/issues/171>`__).

Fixed
^^^^^
- List type with empty list default causes failure (`PyLaia#48
  <https://github.com/jpuigcerver/PyLaia/issues/48>`__).
- Pure dataclass instance default being considered as a subclass type.
- Discard ``init_args`` after ``class_path`` change causes error (`#205
  <https://github.com/omni-us/jsonargparse/issues/205>`__).
- ``fail_untyped=False`` not propagated to subclass ``--*.help`` actions.
- Issues reported by CodeQL.
- Incorrect value when ``Path`` is cast to ``str`` and ``rel_path`` was changed.
- Argument links with target a subclass mixed with other types not working (`#208
  <https://github.com/omni-us/jsonargparse/issues/208>`__).
- Failures when using a sequence type and the default is a tuple.
- Parent parser logger not being forwarded to subcommand and internal parsers.

Changed
^^^^^^^
- Clearer error message for when an argument link targets a subclass and the
  target key does not have ``init_args`` (`lightning#16032
  <https://github.com/Lightning-AI/lightning/issues/16032>`__).
- The ``signatures`` extras now installs the ``typeshed-client`` package.
- ``validators`` package is no longer a dependency.
- Path types are no longer a subclass of ``str``.
- Parsing steps logging now at debug level.
- Discarding ``init_args`` warning changed to log at debug level.
- Removed replacing list instead of append warning.


v4.18.0 (2022-11-29)
--------------------

Added
^^^^^
- AST resolving for defaults with a class instance or a lambda that returns a
  class instance.

Fixed
^^^^^
- ``bool`` values should not be accepted by ``int`` or ``float`` types.
- ``parse_string`` raises ``AttributeError`` when given a simple string.
- Added missing ``return_parser`` deprecation warning when ``CLI`` has
  subcommands.
- Parsing fails for registered types that can't be cast to boolean (`#196
  <https://github.com/omni-us/jsonargparse/issues/196>`__).
- List append not working for ``default_config_files`` set in a subcommand
  subparser (`lightning#15256
  <https://github.com/Lightning-AI/lightning/issues/15256>`__).
- Specifying only the class name through command line not working for
  ``Callable`` with class return type.
- ``init_args`` not discarded for nested subclasses provided through command
  line (`lightning#15796
  <https://github.com/Lightning-AI/lightning/issues/15796>`__).
- Unable to set/get values in ``Namespace`` when key is the same as a method
  name.

Changed
^^^^^^^
- ``CLI`` no longer adds ``--config`` and ``--print_config`` if no arguments
  added to subcommand.
- ``CLI`` now uses the component's docstring short description for subparser
  descriptions.
- Slightly nicer type hint unexpected value error messages, in particular less
  redundancy for ``Union`` types.


v4.17.0 (2022-11-11)
--------------------

Added
^^^^^
- AST resolver now ignores if/elif/else code when condition is a global constant
  (`#187 <https://github.com/omni-us/jsonargparse/issues/187>`__).
- AST resolver support for conditional ``**kwargs`` use in multiple calls (`#187
  comment
  <https://github.com/omni-us/jsonargparse/issues/187#issuecomment-1295141338>`__).

Fixed
^^^^^
- ``str`` type fails to parse value when pyyaml raises ``ConstructorError``
  (`#189 <https://github.com/omni-us/jsonargparse/issues/189>`__).
- ``Namespace`` clone should not deepcopy leaf values (`#187
  <https://github.com/omni-us/jsonargparse/issues/187>`__).
- ``_ActionHelpClassPath`` actions fail to instantiate when base class uses new
  union type syntax.

Changed
^^^^^^^
- Improved help usage and description for ``--print_config``.
- Registering ``pathlib.Path`` types so that they are not shown as subclass
  types.


v4.16.0 (2022-10-28)
--------------------

Added
^^^^^
- Type ``Any`` now parses and instantiates classes when given dict that follows
  subclass specification (`lightning#15115
  <https://github.com/Lightning-AI/lightning/issues/15115>`__).
- Signature methods now accept skipping a number of positionals.
- Callable type hint with return type a class can now be given a subclass which
  produces a callable that returns an instance of the class.
- Support for Python 3.11.

Fixed
^^^^^
- Fail to import on Python 3.7 when typing_extensions not installed (`#178
  <https://github.com/omni-us/jsonargparse/issues/178>`__).
- Crashing when using set typehint with specified dtype (`#183
  <https://github.com/omni-us/jsonargparse/issues/183>`__).

Changed
^^^^^^^
- Using ``set_defaults`` on a config argument raises error and suggests to use
  ``default_config_files`` (`lightning#15174
  <https://github.com/Lightning-AI/lightning/issues/15174>`__).
- Trying to add a second config argument to a single parser raises an exception
  (`#169 <https://github.com/omni-us/jsonargparse/issues/169>`__).


v4.15.2 (2022-10-20)
--------------------

Fixed
^^^^^
- Regression introduced in `6e7ae6d
  <https://github.com/omni-us/jsonargparse/commit/6e7ae6dca41d2bdf081731c042bba9d08b6f228f>`__
  that produced cryptic error message when an invalid argument given (`#172
  <https://github.com/omni-us/jsonargparse/issues/172>`__).
- ``default_env`` not forwarded to subcommand parsers, causing environment
  variable names to not be shown in subcommand help (`lightning#12790
  <https://github.com/Lightning-AI/lightning/issues/12790>`__).
- Cannot override Callable ``init_args`` without passing the ``class_path``
  (`#174 <https://github.com/omni-us/jsonargparse/issues/174>`__).
- Positional subclass type incorrectly adds subclass help as positional.
- Order of types in ``Union`` not being considered.
- ``str`` type fails to parse values of the form ``^\w+: *``.
- ``parse_object`` does not consider given namespace for previous ``class_path``
  values.


v4.15.1 (2022-10-07)
--------------------

Fixed
^^^^^
- ``compute_fn`` of an argument link applied on parse not given subclass default
  ``init_args`` when loading from config.
- Subclass ``--*.help`` option not available when type is a ``Union`` mixed with
  not subclass types.
- Override of ``dict_kwargs`` items from command line not working.
- Multiple subclass ``init_args`` given through command line not being
  considered (`lightning#15007
  <https://github.com/Lightning-AI/lightning/pull/15007>`__).
- ``Union`` types required all subtypes to be supported when expected to be at
  least one subtype supported (`#168
  <https://github.com/omni-us/jsonargparse/issues/168>`__).


v4.15.0 (2022-09-27)
--------------------

Added
^^^^^
- ``set_defaults`` now supports subclass by name and normalization of import path.

Fixed
^^^^^
- Loop variable capture bug pointed out by lgtm.com.
- Issue with discard ``init_args`` when ``class_path`` not a subclass.
- No error shown when arguments given to class group that does not accept
  arguments (`#161 comment
  <https://github.com/omni-us/jsonargparse/issues/161#issuecomment-1256973565>`__).
- Incorrect replacement of ``**kwargs`` when ``*args`` present in parameter resolver.
- Override of ``class_path`` not discarding ``init_args`` when loading from
  config file.
- Invalid values given to the ``compute_fn`` of a argument link applied on parse
  without showing an understandable error message.

Changed
^^^^^^^
- Now ``UUID`` and ``timedelta`` types are registered on first use to avoid
  possibly unused imports.
- json/yaml dump sort now defaults to false for all python implementations.
- ``add_class_arguments`` will not add config load option if no added arguments.


v4.14.1 (2022-09-26)
--------------------

Fixed
^^^^^
- Making ``import_docstring_parse`` a deprecated function only for
  pytorch-lightning backward compatibility.


v4.14.0 (2022-09-14)
--------------------

Added
^^^^^
- Support for ``os.PathLike`` as typehint (`#159
  <https://github.com/omni-us/jsonargparse/issues/159>`__).
- Also show known subclasses in help for ``Type[<type>]``.
- Support for attribute docstrings (`#150
  <https://github.com/omni-us/jsonargparse/issues/150>`__).
- Way to configure parsing docstrings with a single style.

Fixed
^^^^^
- Subclass nested argument incorrectly loaded as subclass config (`#159
  <https://github.com/omni-us/jsonargparse/issues/159>`__).
- Append to list not working for ``default_config_files`` in subcommands (`#157
  <https://github.com/omni-us/jsonargparse/issues/157>`__).


v4.13.3 (2022-09-06)
--------------------

Fixed
^^^^^
- Failure to parse when subcommand has no options (`#158
  <https://github.com/omni-us/jsonargparse/issues/158>`__).
- Optional packages being imported even though not used.
- Append to list not working for ``default_config_files`` (`#157
  <https://github.com/omni-us/jsonargparse/issues/157>`__).


v4.13.2 (2022-08-31)
--------------------

Fixed
^^^^^
- Failure to print help when ``object`` used as type hint.
- Failure to parse init args when type hint is union of str and class.
- Handle change of non-existent file exception type in latest fsspec version.


v4.13.1 (2022-08-05)
--------------------

Fixed
^^^^^
- Regression that caused parse to fail when providing ``init_args`` from command
  line and the subclass default set as a dict.


v4.13.0 (2022-08-03)
--------------------

Added
^^^^^
- Support setting through command line individual dict items without replacing
  (`#133 comment
  <https://github.com/omni-us/jsonargparse/issues/133#issuecomment-1194305222>`__).
- Support ``super()`` with non-immediate method resolution order parameter (`#153
  <https://github.com/omni-us/jsonargparse/issues/153>`__).

Fixed
^^^^^
- Mypy fails to find jsonargparse type hints (`#151
  <https://github.com/omni-us/jsonargparse/issues/151>`__).
- For multiple ``dict_kwargs`` command line arguments only the last one was
  kept.
- Positional ``list`` with subtype causing crash (`#154
  <https://github.com/omni-us/jsonargparse/issues/154>`__).


v4.12.0 (2022-07-22)
--------------------

Added
^^^^^
- Instantiation links now support multiple sources.
- AST resolver now supports ``cls()`` class instantiation in ``classmethod``
  (`#146 <https://github.com/omni-us/jsonargparse/issues/146>`__).
- AST resolver now supports ``pop`` and ``get`` from ``**kwargs``.

Fixed
^^^^^
- `file:///` scheme not working in windows (`#144
  <https://github.com/omni-us/jsonargparse/issues/144>`__).
- Instantiation links with source an entire subclass incorrectly showed
  ``--*.help``.
- Ensure AST-based parameter resolver handles value-less type annotations without error
  (`#148 <https://github.com/omni-us/jsonargparse/issues/148>`__).
- Discarding ``init_args`` on ``class_path`` change not working for ``Union``
  with mixed non-subclass types.
- In some cases debug logs not shown even though ``JSONARGPARSE_DEBUG`` set.

Changed
^^^^^^^
- Instantiation links with source an entire class no longer requires to have a
  compute function.
- Instantiation links no longer restricted to first nesting level.
- AST parameter resolver now only logs debug messages instead of failing (`#146
  <https://github.com/omni-us/jsonargparse/issues/146>`__).
- Documented AST resolver support for ``**kwargs`` use in property.


v4.11.0 (2022-07-12)
--------------------

Added
^^^^^
- ``env_prefix`` property now also accepts boolean. If set to False, no prefix
  is used for environment variable names (`#145
  <https://github.com/omni-us/jsonargparse/pull/145>`__).
- ``link_arguments`` support target being an entire subclass object
  (`lightning#13539
  <https://github.com/Lightning-AI/lightning/discussions/13539>`__).

Fixed
^^^^^
- Method resolution order not working correctly in parameter resolvers (`#143
  <https://github.com/omni-us/jsonargparse/issues/143>`__).

Deprecated
^^^^^^^^^^
- ``env_prefix`` property will no longer accept ``None`` in v5.0.0.


v4.10.2 (2022-07-01)
--------------------

Fixed
^^^^^
- AST resolver fails for ``self._kwargs`` assign when a type hint is added.


v4.10.1 (2022-06-29)
--------------------

Fixed
^^^^^
- "Component not supported" crash instead of no parameters (`#141
  <https://github.com/omni-us/jsonargparse/issues/141>`__).
- Default from ``default_config_files`` not shown in help when argument has no
  default.
- Only ``init_args`` in later config overwrites instead of updates (`#142
  <https://github.com/omni-us/jsonargparse/issues/142>`__).


v4.10.0 (2022-06-21)
--------------------

Added
^^^^^
- Signature parameters resolved by inspecting the source code with ASTs
  (`lightning#11653
  <https://github.com/Lightning-AI/lightning/issues/11653>`__).
- Support init args for unresolved parameters in subclasses (`#114
  <https://github.com/omni-us/jsonargparse/issues/114>`__).
- Allow providing a config with ``init_args`` but no ``class_path`` (`#113
  <https://github.com/omni-us/jsonargparse/issues/113>`__).

Fixed
^^^^^
- ``dump`` with ``skip_default=True`` not working for subclasses without
  ``init_args`` and when a default value requires serializing.
- ``JSONARGPARSE_DEFAULT_ENV`` should have precedence over given value.
- Giving an invalid class path and then init args would print a misleading error
  message about the init arg instead of the class.
- In some cases ``print_config`` could output invalid values. Now a lenient
  check is done while dumping.
- Resolved some issues related to the logger property and reconplogger.
- Single dash ``'-'`` incorrectly parsed as ``[None]``.

Changed
^^^^^^^
- ``dataclasses`` no longer an optional, now an install require on python 3.6.
- Parameters of type ``POSITIONAL_OR_KEYWORD`` now considered ``KEYWORD`` (`#98
  <https://github.com/omni-us/jsonargparse/issues/98>`__).
- Some refactoring mostly related but not limited to the new AST support.
- ``JSONARGPARSE_DEBUG`` now also sets the reconplogger level to ``DEBUG``.
- Renamed the test files to follow the more standard ``test_*.py`` pattern.
- Now ``bool(Namespace())`` evaluates to ``False``.
- When a ``class_path`` is overridden, now only the config values that the new
  subclass doesn't accept are discarded.

Deprecated
^^^^^^^^^^
- ``logger`` property will no longer accept ``None`` in v5.0.0.


v4.9.0 (2022-06-01)
-------------------

Fixed
^^^^^
- ActionsContainer not calling ``LoggerProperty.__init__``.
- For type ``Union[type, List[type]`` when previous value is ``None`` then
  ``--arg+=elem`` should result in a list with single element.

Changed
^^^^^^^
- ``Literal`` options now shown in metavar like choices (`#106
  <https://github.com/omni-us/jsonargparse/issues/106>`__).
- ``tuple`` metavar now shown as ``[ITEM,...]``.
- Required arguments with ``None`` default now shown without brackets in usage.
- Improved description of ``--print_config`` in help.


v4.8.0 (2022-05-26)
-------------------

Added
^^^^^
- Support append to lists both from command line and config file (`#85
  <https://github.com/omni-us/jsonargparse/issues/85>`__).
- New ``register_unresolvable_import_paths`` function to allow getting the
  import paths of objects that don't have a proper ``__module__`` attribute
  (`lightning#13092
  <https://github.com/Lightning-AI/lightning/issues/13092>`__).
- New unit test for merge of config file ``init_args`` when ``class_path`` does
  not change (`#89 <https://github.com/omni-us/jsonargparse/issues/89>`__).

Changed
^^^^^^^
- Replaced custom pre-commit script with a .pre-commit-config.yaml file.
- All warnings are now caught in unit tests.
- Moved ``return_parser`` tests to deprecated tests module.


v4.7.3 (2022-05-10)
-------------------

Fixed
^^^^^
- ``sub_add_kwargs`` not propagated for parameters of final classes.
- New union syntax not working (`#136
  <https://github.com/omni-us/jsonargparse/issues/136>`__).


v4.7.2 (2022-04-29)
-------------------

Fixed
^^^^^
- Make ``import_docstring_parse`` backward compatible to support released
  versions of ``LightningCLI`` (`lightning#12918
  <https://github.com/Lightning-AI/lightning/pull/12918>`__).


v4.7.1 (2022-04-26)
-------------------

Fixed
^^^^^
- Properly catch exceptions when parsing docstrings (`lightning#12883
  <https://github.com/Lightning-AI/lightning/issues/12883>`__).


v4.7.0 (2022-04-20)
-------------------

Fixed
^^^^^
- Failing to parse strings that look like timestamps (`#135
  <https://github.com/omni-us/jsonargparse/issues/135>`__).
- Correctly consider nested mapping type without args as supported.
- New registered types incorrectly considered as class type.

Changed
^^^^^^^
- Final classes now added as group of actions instead of one typehint action.
- ``@final`` decorator now an import from typing_extensions if available.
- Exporting ``ActionsContainer`` to show respective methods in documentation.
- Raise ValueError when logger property given dict with unexpected key.


v4.6.0 (2022-04-11)
-------------------

Added
^^^^^
- Dump option to exclude entries whose value is the same as the default (`#91
  <https://github.com/omni-us/jsonargparse/issues/91>`__).
- Support specifying ``class_path`` only by name for known subclasses (`#84
  <https://github.com/omni-us/jsonargparse/issues/84>`__).
- ``add_argument`` with subclass type now also adds ``--*.help`` option.
- Support shorter subclass command line arguments by not requiring to have
  ``.init_args.``.
- Support for ``Literal`` backport from typing_extensions on python 3.7.
- Support nested subclass ``--*.help CLASS`` options.

Changed
^^^^^^^
- ``class_path``'s on parse are now normalized to shortest form.


v4.5.0 (2022-03-29)
-------------------

Added
^^^^^
- ``capture_parser`` function to get the parser object from a cli function.
- ``dump_header`` property to set header for yaml/jsonnet dumpers (`#79
  <https://github.com/omni-us/jsonargparse/issues/79>`__).
- ``Callable`` type now supports callable classes (`#110
  <https://github.com/omni-us/jsonargparse/issues/110>`__).

Fixed
^^^^^
- Bug in check for ``class_path``, ``init_args`` dicts.
- Module mocks in cli_tests.py.

Changed
^^^^^^^
- Moved argcomplete code from core to optionals module.
- ``Callable`` no longer a simple registered type.
- Import paths are now serialized as its shortest form.
- ``Callable`` default now shown in help as full import path.
- Moved typehint code from core to typehint module.
- Ignore argument links when source/target subclass does not have parameter
  (`#129 <https://github.com/omni-us/jsonargparse/issues/129>`__).
- Swapped order of argument links in help to ``source --> target``.

Deprecated
^^^^^^^^^^
- ``CLI``'s ``return_parser`` parameter will be removed in v5.0.0.


v4.4.0 (2022-03-18)
-------------------

Added
^^^^^
- Environment variables to enable features without code change:
    - ``JSONARGPARSE_DEFAULT_ENV`` to enable environment variable parsing.
    - ``JSONARGPARSE_DEBUG`` to print of stack trace on parse failure.

Fixed
^^^^^
- No error message for unrecognized arguments (`lightning#12303
  <https://github.com/Lightning-AI/lightning/issues/12303>`__).

Changed
^^^^^^^
- Use yaml.CSafeLoader for yaml loading if available.


v4.3.1 (2022-03-01)
-------------------

Fixed
^^^^^
- Incorrect use of ``yaml_load`` with jsonnet parser mode (`#125
  <https://github.com/omni-us/jsonargparse/issues/125>`__).
- Load of subconfigs not correctly changing working directory (`#125
  <https://github.com/omni-us/jsonargparse/issues/125>`__).
- Regression introduced in commit 97e4567 fixed and updated unit test to prevent
  it (`#128 <https://github.com/omni-us/jsonargparse/issues/128>`__).
- ``--print_config`` fails for subcommands when ``default_env=True`` (`#126
  <https://github.com/omni-us/jsonargparse/issues/126>`__).


v4.3.0 (2022-02-22)
-------------------

Added
^^^^^
- Subcommands now also consider parent parser's ``default_config_files``
  (`lightning#11622
  <https://github.com/Lightning-AI/lightning/pull/11622>`__).
- Automatically added group config load options are now shown in the help #121.

Fixed
^^^^^
- Dumper for ``jsonnet`` should be json instead of yaml (`#123
  <https://github.com/omni-us/jsonargparse/issues/123>`__).
- ``jsonnet`` import path not working correctly (`#122
  <https://github.com/omni-us/jsonargparse/issues/122>`__).

Changed
^^^^^^^
- ``ArgumentParser`` objects are now pickleable (`lightning#12011
  <https://github.com/Lightning-AI/lightning/pull/12011>`__).


v4.2.0 (2022-02-09)
-------------------

Added
^^^^^
- ``object_path_serializer`` and ``import_object`` support class methods #99.
- ``parser_mode`` is now a property that when set, propagates to subparsers.
- ``add_method_arguments`` also add parameters from same method of parent
  classes when ``*args`` or ``**kwargs`` present.

Fixed
^^^^^
- Optional Enum types incorrectly adding a ``--*.help`` argument.
- Specific errors for invalid value for ``--*.help class_path``.


v4.1.4 (2022-01-26)
-------------------

Fixed
^^^^^
- Subcommand parsers not using the parent's ``parser_mode``.
- Namespace ``__setitem__`` failing when key corresponds to a nested dict.


v4.1.3 (2022-01-24)
-------------------

Fixed
^^^^^
- String within curly braces parsed as dict due to yaml spec implicit values.


v4.1.2 (2022-01-20)
-------------------

Fixed
^^^^^
- Namespace TypeError with non-str inputs (`#116
  <https://github.com/omni-us/jsonargparse/issues/116>`__).
- ``print_config`` failing on subclass with required arguments (`#115
  <https://github.com/omni-us/jsonargparse/issues/115>`__).


v4.1.1 (2022-01-13)
-------------------

Fixed
^^^^^
- Bad config merging in ``handle_subcommands`` (`lightning#10859
  <https://github.com/Lightning-AI/lightning/issues/10859>`__).
- Unit tests failing with argcomplete>=2.0.0.


v4.1.0 (2021-12-06)
-------------------

Added
^^^^^
- ``set_loader`` function to allow replacing default yaml loader or adding a
  new parser mode.
- ``set_dumper`` function to allow changing default dump formats or adding new
  named dump formats.
- ``parser_mode='omegaconf'`` option to use OmegaConf as a loader, adding
  variable interpolation support.

Fixed
^^^^^
- ``class_from_function`` missing dereference of string return type (`#105
  <https://github.com/omni-us/jsonargparse/issues/105>`__).


v4.0.4 (2021-11-29)
-------------------

Fixed
^^^^^
- Linking of attributes applied on instantiation ignoring compute_fn.
- Show full class paths in ``--*.help`` description to avoid misinterpretation.
- ``--*.help`` action failing when fail_untyped and/or skip is required. (`#101
  <https://github.com/omni-us/jsonargparse/issues/101>`__).
- Raise exception if lazy_instance called with invalid lazy_kwargs.
- Only add subclass defaults on defaults merging (`#103
  <https://github.com/omni-us/jsonargparse/issues/103>`__).
- Strict type and required only on final config check (`#31
  <https://github.com/omni-us/jsonargparse/issues/31>`__).
- instantiate_classes failing for type hints with ``nargs='+'``.
- Useful error message when init_args value invalid.
- Specific error message when subclass dict has unexpected keys.
- Removed unnecessary recursive calls causing slow parsing.


v4.0.3 (2021-11-23)
-------------------

Fixed
^^^^^
- Command line parsing of init_args failing with subclasses without a default.
- get_default failing when destination key does not exist in default config file.
- Fixed issue with empty help string caused by a change in argparse python 3.9.


v4.0.2 (2021-11-22)
-------------------

Fixed
^^^^^
- Specifying init_args from the command line resulting in empty namespace when
  no prior class_path given.
- Fixed command line parsing of class_path and init_args options within
  subcommand.
- lazy_instance of final class leading to incorrect default that includes
  class_path and init_args.
- add_subclass_arguments not accepting a default keyword parameter.
- Make it possible to disable deprecation warnings.


v4.0.0 (2021-11-16)
-------------------

Added
^^^^^
- New Namespace class that natively supports nesting and avoids flat/dict
  conversions.
- python 3.10 is now supported and included in circleci tests.
- Readme changed to use doctest and tests are run in github workflow.
- More type hints throughout the code base.
- New unit tests to increase coverage.
- Include dataclasses extras require for tox testing.
- Automatic namespace to dict for link based on target or compute_fn type.

Fixed
^^^^^
- Fixed issues related to conflict namespace base.
- Fixed the parsing of ``Dict[int, str]`` type (`#87
  <https://github.com/omni-us/jsonargparse/issues/87>`__).
- Fixed inner relative config with for commented tests for parse_env and CLI.
- init_args from default_config_files not discarded when class_path is
  overridden.
- Problems with class instantiation for parameters of final classes.
- dump/save not removing linked target keys.
- lazy_instance not working with torch.nn.Module (`#96
  <https://github.com/omni-us/jsonargparse/issues/96>`__).

Changed
^^^^^^^
- General refactoring and cleanup related to new Namespace class.
- Parsed values from ActionJsonSchema/ActionJsonnet are now dict instead of
  Namespace.
- Removed support for python 3.5 and related code cleanup.
- contextvars package is now an install require for python 3.6.
- Deprecations are now shown as JsonargparseDeprecationWarning.

Deprecated
^^^^^^^^^^
- ArgumentParser's ``parse_as_dict`` option will be removed in v5.0.0.
- ArgumentParser's ``instantiate_subclasses`` method will be removed in v5.0.0.

Removed
^^^^^^^
- python 3.5 is no longer supported.


v3.19.4 (2021-10-04)
--------------------

Fixed
^^^^^
- self.logger undefined on SignatureArguments (`#92
  <https://github.com/omni-us/jsonargparse/issues/92>`__).
- Fix linking for deep targets (`#75
  <https://github.com/omni-us/jsonargparse/pull/75>`__).
- Fix import_object failing with "not enough values to unpack" (`#94
  <https://github.com/omni-us/jsonargparse/issues/94>`__).
- Yaml representer error when dumping unregistered default path type.


v3.19.3 (2021-09-16)
--------------------

Fixed
^^^^^
- add_subclass_arguments with required=False failing on instantiation (`#83
  <https://github.com/omni-us/jsonargparse/issues/83>`__).


v3.19.2 (2021-09-09)
--------------------

Fixed
^^^^^
- add_subclass_arguments with required=False failing when not given (`#83
  <https://github.com/omni-us/jsonargparse/issues/83>`__).


v3.19.1 (2021-09-03)
--------------------

Fixed
^^^^^
- Repeated instantiation of dataclasses (`lightning#9207
  <https://github.com/Lightning-AI/lightning/issues/9207>`__).


v3.19.0 (2021-08-27)
--------------------

Added
^^^^^
- ``save`` now supports saving to an fsspec path (`#86
  <https://github.com/omni-us/jsonargparse/issues/86>`__).

Fixed
^^^^^
- Multifile save not working correctly for subclasses (`#63
  <https://github.com/omni-us/jsonargparse/issues/63>`__).
- ``link_arguments`` not working for subcommands (`#82
  <https://github.com/omni-us/jsonargparse/issues/82>`__).

Changed
^^^^^^^
- Multiple subcommand settings without explicit subcommand is now a warning
  instead of exception.


v3.18.0 (2021-08-18)
--------------------

Added
^^^^^
- Support for parsing ``Mapping`` and ``MutableMapping`` types.
- Support for parsing ``frozenset``, ``MutableSequence`` and ``MutableSet`` types.

Fixed
^^^^^
- Don't discard ``init_args`` with non-changing ``--*.class_path`` argument.
- Don't ignore ``KeyError`` in call to instantiate_classes (`#81
  <https://github.com/omni-us/jsonargparse/issues/81>`__).
- Optional subcommands fail with a KeyError (`#68
  <https://github.com/omni-us/jsonargparse/issues/68>`__).
- Conflicting namespace for subclass key in subcommand.
- ``instantiate_classes`` not working for subcommand keys (`#70
  <https://github.com/omni-us/jsonargparse/issues/70>`__).
- Proper file not found message from _ActionConfigLoad (`#64
  <https://github.com/omni-us/jsonargparse/issues/64>`__).
- ``parse_path`` not parsing inner config files.

Changed
^^^^^^^
- Docstrings no longer supported for python 3.5.
- Show warning when ``--*.class_path`` discards previous ``init_args``.
- Trigger error when ``parse_args`` called with non-string value.
- ActionParser accepts both title and help, title having preference.
- Multiple subcommand settings allowed if explicit subcommand given.


v3.17.0 (2021-07-19)
--------------------

Added
^^^^^
- ``datetime.timedelta`` now supported as a type.
- New function ``class_from_function`` to add signature of functions that
  return an instantiated class.

Fixed
^^^^^
- ``--*.init_args.*`` causing crash when overriding value from config file.


v3.16.1 (2021-07-13)
--------------------

Fixed
^^^^^
- Signature functions not working for classes implemented with ``__new__``.
- ``instantiate_classes`` failing when keys not present in config object.


v3.16.0 (2021-07-05)
--------------------

Added
-----
- ``lazy_instance`` function for serializable class type defaults.
- Support for parsing multiple matched default config files (`#58
  <https://github.com/omni-us/jsonargparse/issues/58>`__).

Fixed
^^^^^
- ``--*.class_path`` and ``--*.init_args.*`` arguments not being parsed.
- ``--help`` broken when default_config_files fail to parse (`#60
  <https://github.com/omni-us/jsonargparse/issues/60>`__).
- Pattern in default_config_files not using sort.


v3.15.0 (2021-06-22)
--------------------

Added
^^^^^
- Decorator for final classes and an is_final_class function to test it.
- Support for final classes as type hint.
- ``add_subclass_arguments`` now supports multiple classes given as tuple.
- ``add_subclass_arguments`` now supports the instantiate parameter.

Fixed
^^^^^
- Parsing of relative paths inside inner configs for type hint actions.


v3.14.0 (2021-06-08)
--------------------

Added
^^^^^
- Method ``instantiate_classes`` that instantiates subclasses and class groups.
- Support for ``link_arguments`` that are applied on instantiation.
- Method ``add_subclass_arguments`` now supports skipping of arguments.
- Added support for Type in type hints (`#59
  <https://github.com/omni-us/jsonargparse/issues/59>`__).

Fixed
^^^^^
- Custom string template to avoid problems with percent symbols in docstrings.


v3.13.1 (2021-06-03)
--------------------

Fixed
^^^^^
- Type hint Any not correctly serializing Enum and registered type values.


v3.13.0 (2021-06-02)
--------------------

Added
^^^^^
- Inner config file support for subclass type hints in signatures and CLI (`#57
  <https://github.com/omni-us/jsonargparse/issues/57>`__).
- Forward fail_untyped setting to nested subclass type hints.

Fixed
^^^^^
- With fail_untyped=True use type from default value instead of Any.
- Registered types and typing types incorrectly considered subclass types.

Changed
^^^^^^^
- Better structure of type hint error messages to ease understanding.


v3.12.1 (2021-05-19)
--------------------

Fixed
^^^^^
- ``--print_config`` can now be given before other arguments without value.
- Fixed conversion of flat namespace to dict when there is a nested empty namespace.
- Fixed issue with get_defaults with default config file and parse_as_dict=False.
- Fixed bug in save which failed when there was an int key.

Changed
^^^^^^^
- ``--print_config`` now only receives a value with ``=`` syntax.
- ``add_{class,method,function,dataclass}_arguments`` now return a list of
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
- Link argument arrow ``<=`` can be confused as less or equal, changed to
  ``<--``.


v3.11.1 (2021-04-30)
--------------------

Fixed
^^^^^
- add_dataclass_arguments not making parameters without default as required (`#54
  <https://github.com/omni-us/jsonargparse/issues/54>`__).
- Removed from signature add methods required option included by mistake.


v3.11.0 (2021-04-27)
--------------------

Added
^^^^^
- CLI now has ``--config`` options at subcommand and subsubcommand levels.
- CLI now adds subcommands with help string taken from docstrings.
- print_config at subcommand level for global config with implicit subcommands.
- New Path_drw predefined type.
- Type hint arguments now support ``nargs='?'``.
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
- ``--*.help`` option being added for non-subclass types.
- Iterable and Sequence types not working for python>=3.7 (`#53
  <https://github.com/omni-us/jsonargparse/issues/53>`__).


v3.10.0 (2021-04-19)
--------------------

Added
^^^^^
- set_defaults method now works for arguments within subcommands.
- CLI set_defaults option to allow overriding of defaults.
- CLI return_parser option to ease inclusion in documentation.
- save_path_content attribute to save paths content on config save.
- New ``link_arguments`` method to derive an argument value from others.
- print_config now includes subclass init_args if class_path given.
- Subclass type hints now also have a ``--*.help`` option.

Changed
^^^^^^^
- Signature parameters whose name starts with "_" are skipped.
- The repr of Path now has the form ``Path_{mode}(``.

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
- Help fails saying required args missing if default config file exists (`#48
  <https://github.com/omni-us/jsonargparse/issues/48>`__).
- ActionYesNo arguments failing when parsing from environment variable (`#49
  <https://github.com/omni-us/jsonargparse/issues/49>`__).


v3.8.0 (2021-03-22)
-------------------

Added
^^^^^
- Path class now supports home prefix '~' (`#45
  <https://github.com/omni-us/jsonargparse/issues/45>`__).
- yaml/json dump kwargs can now be changed via attributes dump_yaml_kwargs and
  dump_json_kwargs.

Changed
^^^^^^^
- Now by default dump/save/print_config preserve the add arguments and argument
  groups order (only CPython>=3.6) (`#46
  <https://github.com/omni-us/jsonargparse/issues/46>`__).
- ActionParser group title now defaults to None if not given (`#47
  <https://github.com/omni-us/jsonargparse/issues/47>`__).
- Add argument with type Enum or type hint giving an action now raises error
  (`#45 <https://github.com/omni-us/jsonargparse/issues/45>`__).
- Parser help now also considers default_config_files and shows which config file
  was loaded (`#47 <https://github.com/omni-us/jsonargparse/issues/47>`__).
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
- The help of ActionParser arguments is now shown in the main help (`#41
  <https://github.com/omni-us/jsonargparse/issues/41>`__).

Fixed
^^^^^
- Use of required in ActionParser parsers not working (`#43
  <https://github.com/omni-us/jsonargparse/issues/43>`__).
- Nested options with names including dashes not working (`#42
  <https://github.com/omni-us/jsonargparse/issues/42>`__).
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
- Tuples with ellipsis are now supported (`#40
  <https://github.com/omni-us/jsonargparse/issues/40>`__).

Fixed
^^^^^
- Using dict as type incorrectly considered as class requiring class_path.
- Nested tuples were not working correctly (`#40
  <https://github.com/omni-us/jsonargparse/issues/40>`__).


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
- Table in readme to ease understanding of extras requires for optional features
  (`#38 <https://github.com/omni-us/jsonargparse/issues/38>`__).

Changed
^^^^^^^
- Save with multifile=True uses file extension to choose json or yaml format.

Fixed
^^^^^
- Better exception message when using ActionJsonSchema and jsonschema not
  installed (`#38 <https://github.com/omni-us/jsonargparse/issues/38>`__).


v3.3.2 (2021-01-22)
-------------------

Fixed
^^^^^
- Changed actions so that keyword arguments are visible in API.
- Fixed save method short description which was copy paste of dump.
- Added missing docstring in instantiate_subclasses method.
- Fixed crash when using ``--help`` and ActionConfigFile not given help string.
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
- Automatic Optional for arguments with default None (`#30
  <https://github.com/omni-us/jsonargparse/issues/30>`__).
- CLI now supports running methods from classes.
- Signature arguments can now be loaded from independent config files (`#32
  <https://github.com/omni-us/jsonargparse/issues/32>`__).
- add_argument now supports enable_path for type based on jsonschema.
- print_config can now be given as value skip_null to exclude null entries.

Changed
^^^^^^^
- Improved description of parser used as standalone and for ActionParser (`#34
  <https://github.com/omni-us/jsonargparse/issues/34>`__).
- Removed ``__cwd__`` and top level ``__path__`` that were not needed.

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
- Support for multiple levels of subcommands (`#29
  <https://github.com/omni-us/jsonargparse/issues/29>`__).
- Default description of subcommands explaining use of ``--help``.


v3.0.1 (2020-12-02)
-------------------

Fixed
^^^^^
- add_class_arguments incorrectly added arguments from ``__call__`` instead
  of ``__init__`` for callable classes.


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
- Parsers now include ``--print_config`` option to dump defaults.
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
