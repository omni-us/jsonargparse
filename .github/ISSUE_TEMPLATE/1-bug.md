---
name: Bug report
about: Create a report to help us improve
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
running it gives. The following snippet templates might help:

1. Using the CLI function

```python
import jsonargparse

# Here define one or more functions or classes
def func1(param1: int, ...):
    ...

# Run the CLI providing the components
jsonargparse.CLI([func1, ...], error_handler=None)
```

2. Manually constructing a parser

```python
import jsonargparse

parser = jsonargparse.ArgumentParser(error_handler=None)
# Here add to the parser only argument(s) relevant to the problem

# If a yaml config is required, it can be included in the same snippet as follows:
import yaml

parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
config = yaml.safe_dump(
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

- jsonargparse version (e.g., 4.8.0):
- Python version (e.g., 3.9):
- How jsonargparse was installed (e.g. `pip install jsonargparse[all]`):
- OS (e.g., Linux):
