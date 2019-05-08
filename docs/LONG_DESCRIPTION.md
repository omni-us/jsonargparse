yamlargparse python module
**************************

This module is an extension to python’s argparse which simplifies
parsing of configuration options from command line arguments, yaml
configuration files, environment variables and hard-coded defaults.

The aim is similar to other projects such as configargparse, yconf and
confuse. The obvious question is, why yet another module similar to
many already existing ones? The answer is simply that none of the
existing projects had the exact features we wanted and after analyzing
the alternatives it seemed simpler to create a new module.


Features
********

* Parsers are configured just like with python’s argparse, thus it
  has a gentile learning curve.

* Not exclusively intended for parsing command line arguments. The
  module has functions to parse environment variables and yaml config
  files.

* Support for nested namespaces which makes it possible to parse
  yaml with non-flat hierarchies.

* Parsing of relative paths within yaml files.

* Configuration settings are overridden based on the following
  precedence.

  * **Parsing command line:** command line arguments (might include
    config file) > environment variables > defaults.

  * **Parsing yaml:** config file values > environment variables >
    defaults.

  * **Parsing environment:** environment variables > defaults.


Using the module
****************

A parser is created just like it is done with argparse. You import the
module, create a parser object and then add arguments to it. A simple
example would be:

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

After creating the parser, you can use it to parse command line
arguments with the "yamlargparse.ArgumentParser.parse_args()"
function, after which you get an object with the parsed values or
defaults available as attributes. For illustrative purposes giving to
"parse_args()" a list of arguments (instead of automatically getting
them from the command line arguments), with the parser from above you
would observe:

   >>> cfg = parser.parse_args(['--opt2', '2.3'])
   >>> cfg.opt1, type(cfg.opt1)
   (0, <class 'int'>)
   >>> cfg.opt2, type(cfg.opt2)
   (2.3, <class 'float'>)


Nested namespaces
*****************

A difference with respect to the basic argparse is that it by using
dot notation in the argument names, you can define a hierarchy of
nested namespaces. So for example you could do the following:

   >>> parser = yamlargparse.ArgumentParser(prog='app')
   >>> parser.add_argument('--lev1.opt1', type=str, default='from default 1')
   >>> parser.add_argument('--lev1.opt2', type=str, default='from default 2')
   >>> cfg = parser.get_defaults()
   >>> cfg.lev1.opt1
   'from default 2'
   >>> cfg.lev1.opt2
   'from default 2'


Environment variables
*********************

The yamlargparse parsers by default also get values from environment
variables. In the case of environment variables, the parser checks
existing variables whose name is of the form "[PROG_][LEV__]*OPT",
that is all in upper case, first only if set the name of the program
followed by underscore and then the argument name replacing dots with
two underscores. Using the parser from the Nested namespaces section
above, in your shell you would set the environment variables as:

   export APP_LEV1__OPT1='from env 1'
   export APP_LEV1__OPT2='from env 2'

Then in python the parser would use these variables, unless overridden
by the command line arguments, that is:

   >>> parser = yamlargparse.ArgumentParser(prog='app')
   >>> parser.add_argument('--lev1.opt1', type=str, default='from default 1')
   >>> parser.add_argument('--lev1.opt2', type=str, default='from default 2')
   >>> cfg = parser.parse_args(['--lev1.opt1', 'from arg 1'])
   >>> cfg.lev1.opt1
   'from arg 1'
   >>> cfg.lev1.opt2
   'from env 2'

There is also the "yamlargparse.ArgumentParser.parse_env()" function
to only parse environment variables, which might be useful for some
use cases in which there is no command line call involved.


YAML configuration files
************************

An important feature of this module is the parsing of yaml files. The
dot notation hierarchy of the arguments (see Nested namespaces) are
used for the expected structure of the yaml files.

When parsing command line arguments, it is possible to add a yaml
configuration file path argument. The yaml file would be read and
parsed in the specific position among the command line arguments, so
the arguments after would override the values from the yaml file.
Again using the parser from the Nested namespaces section above, for
example we could have the following yaml:

   # File: example.yaml
   lev1:
     opt1: from yaml 1
     opt2: from yaml 2

Then in python adding a yaml file argument and parsing some example
arguments, the following would be observed:

   >>> parser = yamlargparse.ArgumentParser(prog='app')
   >>> parser.add_argument('--lev1.opt1', type=str, default='from default 1')
   >>> parser.add_argument('--lev1.opt2', type=str, default='from default 2')
   >>> parser.add_argument('--cfg', action=yamlargparse.ActionConfigFile)
   >>> cfg = parser.parse_args(['--lev1.opt1', 'from arg 1', '--cfg', 'example.yaml', '--lev1.opt2', 'from arg 2'])
   >>> cfg.lev1.opt1
   'from yaml 1'
   >>> cfg.lev1.opt2
   'from arg 2'

There are also functions
"yamlargparse.ArgumentParser.parse_yaml_path()" and
"yamlargparse.ArgumentParser.parse_yaml_string()" to only parse a yaml
file or yaml contained in a string respectively.


Parsing paths
*************

For some use cases it is necessary to parse file paths, checking its
existence and access permissions, but not necessarily opening the
file. Moreover, a file path could be included in a yaml file as
relative with respect to the yaml file’s location. After parsing it
should be easy to access the parsed file path without having to
consider the location of the yaml file. To help in these situations
yamlargparse includes the "ActionPath" class.

For example suppose you have a directory with a configuration file
"app/config.yaml" and some data "app/data/info.db". The contents of
the yaml file is the following:

   # File: config.yaml
   databases:
     info: data/info.db

To create a parser that checks that the value of "databases.info"
exists and is readable, the following could be done:

   >>> parser = yamlargparse.ArgumentParser(prog='app')
   >>> parser.add_argument('--databases.info', action=yamlargparse.ActionPath(mode='r'))
   >>> cfg = parser.parse_yaml('app/config.yaml')

After parsing it is possible to get both the original relative path as
included in the yaml file, or the corresponding absolute path:

   >>> cfg.databases.info(absolute=False)
   'data/info.db'
   >>> cfg.databases.info()
   'YOUR_CWD/app/data/info.db'

Likewise directories can also be parsed by including in the mode the
"'d'" flag, e.g. "ActionPath(mode='drw')".


Comparison operators
********************

It is quite common that when parsing a number, its range should be
limited. To ease these cases the module includes the
"ActionOperators". Some examples of arguments that can be added using
this action are the following:

   # Larger than zero
   parser.add_argument('--op1', action=yamlargparse.ActionOperators(expr=('>', 0)))
   # Between 0 and 10
   parser.add_argument('--op2', action=yamlargparse.ActionOperators(expr=[('>=', 0), ('<=', 10)]))


API Reference
*************

class yamlargparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=<class 'argparse.HelpFormatter'>, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True)

   Bases: "argparse.ArgumentParser"

   Extension to python’s argparse which simplifies parsing of
   configuration options from command line arguments, yaml
   configuration files, environment variables and hard-coded defaults.

   parse_args(*args, env=True, nested=True, **kwargs)

      Parses command line argument strings.

      All the arguments from argparse.ArgumentParser.parse_args are
      supported. Additionally it accepts:

      Parameters:
         * **env** (*bool*) – Whether to merge with the parsed
           environment.

         * **nested** (*bool*) – Whether the namespace should be
           nested.

      Returns:
         An object with all parsed values as nested attributes.

      Return type:
         SimpleNamespace

   parse_yaml_path(yaml_path, env=True, defaults=True, nested=True)

      Parses a yaml file given its path.

      Parameters:
         * **yaml_path** (*str*) – Path to the yaml file to parse.

         * **env** (*bool*) – Whether to merge with the parsed
           environment.

         * **defaults** (*bool*) – Whether to merge with the
           parser’s defaults.

         * **nested** (*bool*) – Whether the namespace should be
           nested.

      Returns:
         An object with all parsed values as nested attributes.

      Return type:
         SimpleNamespace

   parse_yaml_string(yaml_str, env=True, defaults=True, nested=True)

      Parses yaml given as a string.

      Parameters:
         * **yaml_str** (*str*) – The yaml content.

         * **env** (*bool*) – Whether to merge with the parsed
           environment.

         * **defaults** (*bool*) – Whether to merge with the
           parser’s defaults.

         * **nested** (*bool*) – Whether the namespace should be
           nested.

      Returns:
         An object with all parsed values as attributes.

      Return type:
         SimpleNamespace

   dump_yaml(cfg)

      Generates a yaml string for a configuration object.

      Parameters:
         **cfg** (*SimpleNamespace | dict*) – The configuration object
         to dump.

      Returns:
         The configuration in yaml format.

      Return type:
         str

   parse_env(env=None, defaults=True, nested=True)

      Parses environment variables.

      Parameters:
         * **env** (*object*) – The environment object to use, if
           None *os.environ* is used.

         * **defaults** (*bool*) – Whether to merge with the
           parser’s defaults.

         * **nested** (*bool*) – Whether the namespace should be
           nested.

      Returns:
         An object with all parsed values as attributes.

      Return type:
         SimpleNamespace

   get_defaults(nested=True)

      Returns a namespace with all default values.

      Parameters:
         **nested** (*bool*) – Whether the namespace should be nested.

      Returns:
         An object with all default values as attributes.

      Return type:
         SimpleNamespace

   add_argument_group(*args, name=None, **kwargs)

      Adds a group to the parser.

      All the arguments from
      argparse.ArgumentParser.add_argument_group are supported.
      Additionally it accepts:

      Parameters:
         **name** (*str*) – Name of the group. If set the group object
         will be included in the parser.groups dict.

      Returns:
         The group object.

   check_config(cfg, skip_none=False)

      Checks that the content of a given configuration object conforms
      with the parser.

      Parameters:
         * **cfg** (*SimpleNamespace | dict*) – The configuration
           object to check.

         * **skip_none** (*bool*) – Whether to skip checking of
           values that are None.

   static merge_config(cfg_from, cfg_to)

      Merges the first configuration into the second configuration.

      Parameters:
         * **cfg_from** (*SimpleNamespace | dict*) – The
           configuration from which to merge.

         * **cfg_to** (*SimpleNamespace | dict*) – The configuration
           into which to merge.

      Returns:
         The merged configuration with same type as cfg_from.

      Return type:
         SimpleNamespace | dict

class yamlargparse.ActionConfigFile(**kwargs)

   Bases: "argparse.Action"

   Action to indicate that an argument is a configuration file.

class yamlargparse.ActionYesNo(**kwargs)

   Bases: "argparse.Action"

   Paired action –opt, –no_opt to set True or False respectively.

class yamlargparse.ActionOperators(**kwargs)

   Bases: "argparse.Action"

   Action to restrict a number range with comparison operators.

   Parameters:
      * **expr** (*tuple** or **list of tuples*) – Pairs of
        operators (> >= < <= == !=) and reference values, e.g. [(‘>=’,
        1),…].

      * **join** (*str*) – How to combine multiple comparisons, must
        be ‘or’ or ‘and’ (default=’and’).

      * **numtype** (*type*) – The value type, either int or float
        (default=int).

class yamlargparse.ActionPath(**kwargs)

   Bases: "argparse.Action"

   Action to check and store a file path.

   Parameters:
      **mode** (*str*) – The required type and access permissions
      among [drwx] as a keyword argument, e.g. ActionPath(mode=’drw’).

class yamlargparse.Path(path, mode='r', cwd=None)

   Bases: "object"

   Stores a (possibly relative) path and the corresponding absolute
   path.

   When an object is created it is checked that: the path exists,
   whether it is a file or directory and has the required access
   permissions. The absolute path of can be obtained without having to
   remember the working directory from when the object was created.

   Parameters:
      * **path** (*str*) – The path to check and store.

      * **mode** (*str*) – The required type and access permissions
        among [drwx].

      * **cwd** (*str*) – Working directory for relative paths. If
        None, then os.getcwd() is used.

   Args called:
      absolute (bool): If false returns the original path given,
      otherwise the corresponding absolute path.

yamlargparse.raise_(ex)

   Raise that works within lambda functions.

   Parameters:
      **ex** (*Exception*) – The exception object to raise.


License
*******

The MIT License (MIT)

Copyright (c) 2019-present, Mauricio Villegas <mauricio@omnius.com>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
“Software”), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


