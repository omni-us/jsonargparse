.. include:: README.rst

.. include:: ../CONTRIBUTING.rst

.. _api-ref:

API Reference
=============

Even though jsonargparse has several internal modules, users are expected to
only import from the main ``jsonargparse`` or ``jsonargparse.typing``. This
allows doing internal refactorings without affecting dependants. Only objects
explicitly exposed in ``jsonargparse.__init__.__all__`` and
``jsonargparse.typing.__all__`` can be considered public.


jsonargparse
------------
.. automodule:: jsonargparse

jsonargparse.typing
-------------------
.. automodule:: jsonargparse.typing


Index
=====

* :ref:`changelog`
* :ref:`license`
* :ref:`genindex`
