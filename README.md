sprokit
=======

Sprokit is the "Stream Processing Toolkit", a library aiming to make processing
a stream of data with various algorithms easy. It supports divergent and
convergent data flows with synchronization between them, connection type
checking, all with full, first-class Python bindings.

Sprokit tends towards making enforced checks (though escape hatches are
available) to avoid common errors in code. These checks allow for code to make
more assumptions while threading or sharing data. Sprokit "pipelines" consist
of "processes" (which have data "ports") and connections between the ports.
Each data port describes the data it is expecting in with a "type" string and a
set of flags describing what it expects of the data. These flags describe the
data sharing policy for the ports so that the pipeline can detect that two
threads may be doing improper data sharing. Once a pipeline is constructed, a
"scheduler" runs the pipeline by telling which processes should run when. There
is also a domain-specific language for its pipelines so that combining
processes together requires no code to write or compile.

Mailing Lists
=============

User's list:

    http://public.kitware.com/cgi-bin/mailman/listinfo/sprokit-users
    sprokit-users@public.kitware.com

Developer's list:

    http://public.kitware.com/cgi-bin/mailman/listinfo/sprokit-developers
    sprokit-developers@public.kitware.com

Development
===========

When developing on sprokit, please keep to the prevailing style of the code.
Some guidelines to keep in mind for different languages in the codebase are as
follows:

CMake
-----

  * 2-space indentation
  * Lowercase for private variables
  * Uppercase for user-controlled variables
  * Prefer functions over macros
    - They have variable scoping and debugging them is much easier
  * Prefer ``foreach (IN LISTS)`` and ``list(APPEND)``
  * Prefer ``sprokit_configure_file`` over ``configure_file`` when possible to
    avoid adding dependencies to the configure step
  * Use the ``sprokit_`` wrappers of common commands (e.g., ``add_library``,
    ``add_test``, etc.) as they automatically Do The Right Thing with
    installation, compile flags, build locations, and more)
  * Quote *all* paths and variable expansions unless list expansion is required
    (usually in command arguments or optional arguments)

C++
---

  * 2-space indentation
  * Use lowercase with underscores for symbol names
  * Store intermediate values into local ``const`` variables so that they are
    easily available when debugging
  * There is no fixed line length, but keep it reasonable
  * Default to using ``const`` everywhere
  * All modifiers of a type go *after* the type (e.g., ``char const*``, not
    ``const char*``)
  * Export symbols (or import them if possible)
  * Use braces around all control (even single-line if) blocks
  * Use typedefs
  * Use exceptions and return values, not error codes and output parameters
    - This allows for chaining functions, works with ``<algorithm>`` better,
      and allows more variables to be ``const``

Python
------

  * Follow PEP8
  * When catching exceptions, catch the type then use ``sys.exc_info()`` so
    that it works in Python versions from 2.4 to 3.3
  * No metaclasses; they don't work with the same syntax in Python2 and Python3
  * Avoid 'with' since it doesn't work in Python 2.4

Testing
-------

Generally, all new code should come with tests. The goal is sustained 95%
coverage and higher (due to impossible-to-generically-create corner cases such
as files which are readable, but error out in the middle). Tests should be
grouped into a single executable for each class, group of cooperating classes
(e.g., process tests), or higher-level use case (e.g., running a pipeline). In
C++, use the ``TEST_`` macros which will hook into the testing infrastructure
automatically and in Python, name functions so that they start with ``test_``
and they will be picked up automatically.

Submitting Patches
==================

Patches may be sent to the developer's list with the standard git format-patch
and git send-email combo. Here's an example submission:

    # Base the new branch off of master.
    % git checkout -b dev/topic master
    # Edit code.
    % $EDITOR
    # Make commits.
    % git commit
    # When done, create a set of patches with a cover letter.
    % git format-patch --output-directory patches/topic \
        --signoff --cover-letter master..
    # Describe the branch in the cover letter.
    % $EDITOR patches/topic/0000*
    # Send the patches to the list.
    % git send-email --to sprokit-developers@public.kitware.com \
        --no-chain-reply-to patches/topic/*.patch
