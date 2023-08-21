Contributing
============

Contributions to jsonargparse are very welcome. There are multiple ways for
people to help and contribute, among them:

- Star ⭐ the github project `<https://github.com/omni-us/jsonargparse/>`__.
- `Sponsor 🩷 <https://github.com/sponsors/mauvilsa>`__ its maintenance and
  development.
- Spread the word in your community about the features you like from
  jsonargparse.
- Help others to learn how to use jsonargparse by creating tutorials, such as
  blog posts and videos.
- Become active in existing github issues and pull requests.
- Create `issues <https://github.com/omni-us/jsonargparse/issues>`__ for
  reporting bugs and proposing improvements.
- Create `pull requests <https://github.com/omni-us/jsonargparse/pulls>`__ with
  documentation improvements, bug fixes or new features.

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

The ``pre-push`` stage runs several hooks (tests, doctests, mypy, coverage) that
take some time. These are intended to let developers know problems which must be
resolved for any pull request to be considered for merging. If you wish to push
without running these hooks, use the command ``git push --no-verify``.

Documentation
-------------

To build the documentation run:

.. code-block:: bash

    sphinx-build sphinx sphinx/_build sphinx/*.rst

To view the built documentation, open the file ``sphinx/_build/index.html`` in a
browser.

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
    python -m jsonargparse_tests             # Run tests on installed package (requires pytest and pytest-subtests)
    pre-commit run -a --hook-stage pre-push  # Run pre-push git hooks (tests, doctests, mypy, coverage)
