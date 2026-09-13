"""
Vendored copy of liberator 0.1.1 (https://gitlab.kitware.com/python/liberator).

Static code extraction, used by the vendored torch_liberator exporter to pull
a model's source into a standalone deploy. Like torch_liberator this used to be
a pip dependency rewritten in place after install, by a ReplaceStringInFile in
cmake/custom_install_viame.cmake that silently became a no-op whenever upstream
touched the line it matched. The fix is applied here instead:

    * core.py -- ub.ensure_unicode() was removed from ubelt.

Imports are relative, so this is reachable as
``viame.pytorch.netharn.liberator`` and not as a top-level module. Its
third-party imports are all already VIAME dependencies: astunparse and ubelt at
module level, pyflakes, pygtrie, kwarray and rich lazily. parso (an
experimental writer, plus starfinder) and xdoctest (a CLI helper) are reachable
only from paths VIAME does not use, and xdoctest is not installed.

Upstream links:
+---------------+---------------------------------------------+
| Github        | https://gitlab.kitware.com/python/liberator |
+---------------+---------------------------------------------+
| Pypi          | https://pypi.org/project/liberator          |
+---------------+---------------------------------------------+
| ReadTheDocs   | https://liberator.readthedocs.io/en/latest/ |
+---------------+---------------------------------------------+
"""


__mkinit__ = """
mkinit -m liberator
mkinit -m liberator --diff
mkinit -m liberator --diff --help
"""

__version__ = '0.1.1'

from . import core as closer
from .core import (Closer,)

__explicit__ = ['Closer', 'closer']  # fixme, mkinit is not respecting this

# ^^^ Backwards compatibility ^^^


from . import core
from .core import (Liberator,)

__all__ = ['Closer', 'Liberator', 'closer', 'core']
