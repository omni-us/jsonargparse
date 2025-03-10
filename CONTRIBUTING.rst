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

    ``.pre-commit-config.yaml`` is configured to run the hooks using python
    3.12. Thus, you need to have python 3.12 installed for pre-commit to work.

The ``pre-push`` stage runs several hooks (tests, doctests, mypy, coverage) that
take some time. These hooks are intended to inform developers of issues that
must be resolved before any pull request can be considered for merging. If you
wish to push without running these hooks, use the command ``git push
--no-verify``.

Formatting of the code is done automatically by pre-commit. If some pre-commit
hooks fail and you decide to skip them, the formatting will be automatically
applied by a github action in pull requests.

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
the package, thus can be run in a production system. Also pre-commit runs some
additional tests.

.. code-block:: bash

    tox                                      # Run tests using tox on available python versions
    pytest                                   # Run tests using pytest on the python of the environment
    pytest --cov                             # Run tests and generate coverage report
    python -m jsonargparse_tests             # Run tests on installed package (requires pytest and pytest-subtests)
    pre-commit run -a --hook-stage pre-push  # Run pre-push git hooks (tests, doctests, mypy, coverage)

Coverage
--------

For a nice html test coverage report, run:

.. code-block:: bash

    pytest --cov --cov-report=html

Then open the file ``htmlcov/index.html`` in a browser.

To get a full coverage report, you need to install all supported python
versions, and then:

.. code-block:: bash

    rm -fr jsonargparse_tests/.coverage jsonargparse_tests/htmlcov
    tox -- --cov=../jsonargparse --cov-append
    cd jsonargparse_tests
    coverage html

Then open the file ``jsonargparse_tests/htmlcov/index.html`` in a browser.

Pull requests
-------------

When creating a pull request, it is recommended that in your fork, create a
specific branch for the changes you want to contribute, instead of using the
``main`` branch.

The required tasks to do for a pull request, are listed in
`PULL_REQUEST_TEMPLATE.md
<https://github.com/omni-us/jsonargparse/blob/main/.github/PULL_REQUEST_TEMPLATE.md>`__.

One of the tasks is adding a changelog entry. For this, note that this project
uses semantic versioning. Depending on whether the contribution is a bug fix or
a new feature, the changelog entry would go in a patch or minor release. The
changelog section for the next release does not have a definite date, for
example:

.. code-block::

    v4.28.0 (2024-03-??)
    --------------------

    Added
    ^^^^^
    -

If no such section exists, just add it. Have a look at previous releases to
decide under which subsection the new entry should go. If you are unsure, ask in
the pull request.

Please don't open pull requests with breaking changes unless this has been
discussed and agreed upon in an issue.
