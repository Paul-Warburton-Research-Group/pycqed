.. PyCQED documentation master file, created by
   sphinx-quickstart on Mon May 30 15:27:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. mdinclude:: ../README.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Circuit QED Primer
==================

Here we briefly introduce Circuit Quantum Electrodynamics (CQED) for simulating Josephson Junction (JJ) circuits :cite:`Vool2017,Kerman2017`.

.. If :math:`\sigma_{1}` equals :math:`\sigma_{2}` then etc, etc.

Transformations
===============

Here we discuss the use of mode transformations for making Born-Oppenheimer (BO) approximations and reducing Hilbert space size.

Ising Parameters
================

Here we discuss methods of projecting full circuit Hamiltonians onto simpler Ising systems :cite:`Consani2019`.

How the Code Works
==================

Here we describe how the code builds circuit Hamiltonians and how parameter sweeps are performed.

Module circuit_graph
====================

.. automodule:: pycqed.circuit_graph
   :members:
   :undoc-members:

Module symbolic_system
======================

.. automodule:: pycqed.symbolic_system
   :members:
   :undoc-members:

Module numerical_system
=======================

.. automodule:: pycqed.numerical_system
   :members:
   :undoc-members:

Module parameters
=================

.. automodule:: pycqed.parameters
   :members:
   :undoc-members:

Module units
============

.. automodule:: pycqed.units
   :members:
   :undoc-members:

Module dataspec
===============

.. automodule:: pycqed.dataspec
   :members:
   :undoc-members:

Module util
===========

.. automodule:: pycqed.util
   :members:
   :undoc-members:

Module text2latex
=================

.. automodule:: pycqed.text2latex
   :members:
   :undoc-members:

Module physical_constants
=========================

.. automodule:: pycqed.physical_constants
   :members:
   :undoc-members:

References
==========

.. bibliography:: refs.bib
   :style: plain
