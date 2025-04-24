---
name: Regression report
about: Report for something that used to work but doesn't anymore
title: ''
labels: bug
assignees: ''
---

<!-- If you like this project, please ⭐ star it https://github.com/omni-us/jsonargparse/ -->

## 🕰️ Regression report

<!-- A clear and concise description of the regression. -->

### To reproduce

<!--
Please include a code snippet that reproduces the regression. Make sure that it
is a script that can be used to run git bisect. This means that when code works
correctly, the script terminates with zero exit code. When it fails an exception
is raised. The following snippet templates might help. Replace "..." with actual
implementation details.

1. a) Using the auto_cli function

```python
#!/usr/bin/env python3

from jsonargparse import auto_cli

# Here define one or more functions or classes
def func1(param1: int, ...):
    ...

# Run the CLI providing the components and arguments
auto_cli(
    [func1, ...],
    args=["--param1=value1", ...],
    exit_on_error=False,
)
```

1. b) Manually constructing a parser

```python
#!/usr/bin/env python3

from jsonargparse import ArgumentParser

parser = ArgumentParser(exit_on_error=False)
# Here add to the parser only argument(s) relevant to the problem

# If a config is required, it can be included in the same snippet as follows:
import json

parser.add_argument("--config", action="config")
config = json.dumps(
    {
        "key1": "val1",
    }
)

# If the problem is when parsing arguments
result = parser.parse_args([f"--config={config}", "--key2=val2", ...])

# If the problem is in class instantiation
parser.instantiate_classes(result)
```

2. Preferably, run git bisect and include in the report the git commit hash that
caused the regression. This would be like:

```
chmod +x regression.py  # make script executable
pip3 install -e .  # install as editable
git bisect start
git bisect bad <version-tag-latest>
git bisect good <version-tag-known-good>
git bisect run ./regression.py
```
-->

### Prior behavior

<!-- Please describe the prior behavior in detail, and contrast it with the behavior you are currently observing. -->

### Environment

<!-- Fill in the list below. -->

- jsonargparse version:
- Python version:
- How jsonargparse was installed:
- OS:
