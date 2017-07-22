Contributing to KWIVER
======================

Topic branches should be named starting with a 'dev/' prefix to distinguish
them from integration branches like 'master' and 'release'.

When developing KWIVER, please keep to the prevailing style of the code.
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
  * Prefer ``kwiver_configure_file`` over ``configure_file`` when possible to
    avoid adding dependencies to the configure step
  * Use the ``kwiver_`` wrappers of common commands (e.g., ``add_library``,
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

Testing
-------

Generally, all new code should come with tests. The goal is sustained
95% coverage and higher (due to impossible-to-generically-create
corner cases such as files which are readable, but error out in the
middle). Tests should be grouped into a single executable for each
class, group of cooperating classes (e.g., types tests), or
higher-level use case. In C++, use the ``TEST_`` macros which will
hook into the testing infrastructure automatically and in Python, name
functions so that they start with ``test_`` and they will be picked up
automatically.
