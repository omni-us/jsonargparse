---
name: Bug report
about: Report for something that doesn't work as expected
title: ''
labels: bug
assignees: ''
---

<!-- If you like this project, please ⭐ star it https://github.com/omni-us/jsonargparse/ -->

## 🐛 Bug report

<!-- A clear and concise description of the bug. -->

### To reproduce

<!--
Please include a code snippet that reproduces the bug and the output that
running it gives. The following snippet templates might help. Replace "..." with
actual implementation details.

1. Using the auto_cli function

```python
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

2. Manually constructing a parser

```python
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

# Preferable that the command line arguments are given to the parse_args call
result = parser.parse_args([f"--config={config}", "--key2=val2", ...])

# If the problem is in the parsed result, print it to stdout
print(parser.dump(result))

# If the problem is in class instantiation
parser.instantiate_classes(result)
```
-->

### Expected behavior

<!-- Describe how the behavior or output of the reproduction snippet should be. -->

### Environment

<!-- Fill in the list below. -->

- jsonargparse version:
- Python version:
- How jsonargparse was installed:
- OS:
