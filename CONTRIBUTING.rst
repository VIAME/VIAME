======================
Contributing to KWIVER
======================

Pull Requests
=============

Integration Branches
--------------------

There a two primary integration branches named ``release`` and ``master``.
Generally, the ``release`` branch contains that last versioned stable release
plus a few patches in preparation for the next patch release.  The ``master``
branch contains new features and API changes since the last release and is
preparing for the next major or minor versioned release.

If your PR is a bug fix, unit testing improvement, or documentation enhancement
that applies to the last release, please branch off of ``release`` and submit
your PR to the ``release`` branch. If your PR is a new feature or bug fix
that applies to ``master`` but not to ``release`` then submit your PR to the
``master`` branch.  Any PR accepted in ``release`` is also accepted into
``master``, but not vice versa.

Release Notes
-------------

When making a PR, the topic branch should almost always include a commit which
updates the relevant release notes text file found in `<doc/release-notes>`_.
The relevant release notes file differs depending on whether you are targeting
``release`` or ``master``, but generally this is the file with the highest
version number assuming you have branched from the correct location.  That is,
the ``release`` branch will not contain the later-versioned release notes file
found on the ``master`` branch.

The changes to the release notes files should, at a very high level, describe
the changes that the PR is introducing.  Usually you would add one or more
bullet points to describe the changes.  Bullet points should be entered in
the appropriate section.  There are sections for Updates (enhancements) as
well as Fixes.  There are subsection for different components of the code.
Most changes on the ``release`` branch go under Fixes, and most changes on
the ``master`` branch go under Updates.

If the code change on the topic branch impacts an existing release note
then the release note should be updated.  If the PR is to fix a bug
on the master branch that was introduced since the last release, then this
should **not** be documented in the Fixes section of the release notes
because the bug itself was never released.  This is one of the few cases
where release notes updates are not required on a PR.

Branch Naming
-------------

Topic branches should be named starting with a ``dev/`` prefix to distinguish
them from integration branches like ``master`` and ``release``.

Code Review
-----------

Pull requests are reviewed by one or more of the core KWIVER maintainers
using the Github tools for discussions.  Maintainers should not merge
a PR until it conforms to the requirements described here (e.g.
coding style, release notes, etc.) and it is confirmed that the code
has sufficient unit tests and does not break any existing unit tests.


Coding Style
============

When developing KWIVER, please keep to the prevailing style of the code.
Some guidelines to keep in mind for different languages in the codebase are as
follows:

CMake
-----

* 2-space indentation

* Lowercase for private variables

* Uppercase for user-controlled variables

* Prefer functions over macros

  * They have variable scoping and debugging them is much easier

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

  * This allows for chaining functions, works with ``<algorithm>`` better,
    and allows more variables to be ``const``

Python
------

* Follow PEP8

* When catching exceptions, catch the type then use ``sys.exc_info()`` so
  that it works in Python versions from 2.4 to 3.3

* No metaclasses; they don't work with the same syntax in Python2 and Python3


Testing
=======

Generally, all new code should come with tests. The goal is sustained
95% coverage and higher (due to impossible-to-generically-create
corner cases such as files which are readable, but error out in the
middle). Tests should be grouped into a single executable for each
class, group of cooperating classes (e.g., types tests), or
higher-level use case. In C++, use the ``TEST_`` macros which will
hook into the testing infrastructure automatically and in Python, name
functions so that they start with ``test_`` and they will be picked up
automatically.
