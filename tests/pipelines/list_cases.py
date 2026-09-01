"""Print every discoverable pipeline case, one per line, for ctest generation.

This deliberately reuses cases.discover(), the same entry point the pytest
modules parametrize over, so the generated ctest entries cannot drift from the
cases pytest will actually collect. Output is::

    <category>\t<case id>\t<skip reason or empty>

Discovery reads the install tree, so this must run after install; that is why
it is invoked at ctest time rather than at configure time.
"""

import sys
from pathlib import Path

# cases.py uses package-relative imports, so it has to be loaded as
# tests.pipelines.cases even when this file is run as a plain script, which is
# how the ctest discovery step invokes it.
if __package__:
    from .cases import discover
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.pipelines.cases import discover


def main(categories):
    for category in categories:
        for case in discover(category):
            # Semicolons and tabs would corrupt the CMake list parsing on
            # the consuming side, so they are neutralised here.
            skip = (case.skip or "").replace("\t", " ").replace("\n", " ")
            skip = skip.replace(";", ",")
            print(f"{category}\t{case.id}\t{skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
