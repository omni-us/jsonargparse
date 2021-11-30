Contributing
============

Contributions to jsonargparse are very welcome, be it just to create `issues
<https://github.com/omni-us/jsonargparse/issues>`_ for reporting bugs and
proposing enhancements, or more directly by creating `pull requests
<https://github.com/omni-us/jsonargparse/pulls>`_.

If you intend to work with the source code, note that this project does not
include any :code:`requirements.txt` file. This is by intention. To make it very
clear what are the requirements for different use cases, all the requirements of
the project are stored in the file :code:`setup.cfg`. The basic runtime
requirements are defined in section :code:`[options]` in the
:code:`install_requires` entry. All extras requires for optional features listed
in :ref:`installation` are stored in section :code:`[options.extras_require]`.
Also there are :code:`test`, :code:`test_no_urls`, :code:`dev` and :code:`doc`
entries in the same :code:`[options.extras_require]` section which lists
requirements for testing, development and documentation building.

The recommended way to work with the source code is the following. First clone
the repository, then create a virtual environment, activate it and finally
install the development requirements. More precisely the steps are:

.. code-block:: bash

    git clone https://github.com/omni-us/jsonargparse.git
    cd jsonargparse
    virtualenv -p python3 venv
    . venv/bin/activate

The crucial step is installing the requirements which would be done by running:

.. code-block:: bash

    pip install -e ".[dev,all]"

Running the unit tests can be done either using using `tox
<https://tox.readthedocs.io/en/stable/>`__ or the :code:`setup.py` script. The
unit tests are also installed with the package, thus can be run in a production
system.

.. code-block:: bash

    tox                            # Run tests using tox
    ./setup.py test_coverage       # Run tests and generate coverage report
    python3 -m jsonargparse_tests  # Run tests for installed package
