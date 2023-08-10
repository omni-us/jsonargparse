Contributing
============

Contributions to jsonargparse are very welcome, be it just to create `issues
<https://github.com/omni-us/jsonargparse/issues>`_ for reporting bugs and
proposing enhancements, or more directly by creating `pull requests
<https://github.com/omni-us/jsonargparse/pulls>`_.

Development environment
-----------------------

If you intend to work with the source code, note that this project does not
include any ``requirements.txt`` file. This is by intention. To make it very
clear what are the requirements for different use cases, all the requirements of
the project are stored in the file ``pyproject.toml``. The basic runtime
requirements are defined in ``dependencies``. Requirements for optional features
are stored in ``[project.optional-dependencies]``. Also in the same section
there are requirements for testing, development and documentation building:
``test``, ``dev`` and ``doc``.

The recommended way to work with the source code is the following. First clone
the repository, then create a virtual environment, activate it and finally
install the development requirements. More precisely the steps are:

.. code-block:: bash

    git clone https://github.com/omni-us/jsonargparse.git
    cd jsonargparse
    python -m venv venv
    . venv/bin/activate

The crucial step is installing the requirements which would be done by running:

.. code-block:: bash

    pip install -e ".[dev,all]"

pre-commit
----------

Please also install the `pre-commit <https://pre-commit.com/>`__ git hook
scripts so that unit tests and code checks are automatically run locally. This
is done as follows:

.. code-block:: bash

    pre-commit install

.. note::

    The ``.pre-commit-config.yaml`` file was changed such that some hooks are
    now run on ``pre-push``. If you have an old development environment, please
    run ``pre-commit install`` again to update the git hooks.

Documentation
-------------

To build the documentation run:

.. code-block:: bash

    sphinx-build sphinx sphinx/_build sphinx/index.rst

Then to see the built documentation, open the file ``sphinx/_build/index.html``
in a browser.

Tests
-----

Running the unit tests can be done either using using `pytest
<https://docs.pytest.org/>`__ or `tox
<https://tox.readthedocs.io/en/stable/>`__. The tests are also installed with
the package, thus can be run in a production system.

.. code-block:: bash

    tox                                      # Run tests using tox on available python versions
    pytest                                   # Run tests using pytest on the python of the environment
    pytest --cov                             # Run tests and generate coverage report
    python -m jsonargparse_tests             # Run tests for installed package (requires pytest and pytest-subtests)
    pre-commit run -a --hook-stage pre-push  # Run pre-push git hook (tests, doctests, mypy, coverage)
