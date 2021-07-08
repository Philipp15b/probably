======================
Probably Documentation
======================

**Probably** is a Python package for parsing, type-checking, and analyzing probabilistic programs written in the pGCL language.

For more information, visit `https://github.com/Philipp15b/probably <https://github.com/Philipp15b/probably>`_.

Features:

* pGCL language support: Constants, variables, static types (boolean, (bounded) natural numbers, real numbers), literals, unary and binary operators, while loops, conditional if statements, assignments, probabilistic choices.
* Weakest pre-expectation calculation for loop-free and linear pGCL programs.

  * Translation from general expectations to linear expectations and expectations in summation normal form.
* Program AST traversal and modification using iterators over mutable references.
* General algorithm for variable substitution in program expressions with substitution expressions.

-----

.. git_commit_detail::
    :branch:
    :commit:
.. git_changelog::
    :revisions: 3

-----


API Documentation
-----------------

.. note::

    You want a quickstart? Jump over to the :doc:`the pGCL module documentation <pgcl>`!

.. toctree::

    pgcl
    pysmt
    analysis
    util


Command-Line Interface
----------------------
.. automodule:: probably.cmd
