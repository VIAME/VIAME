Example external python plugin.

A python package that adds an image filter algorithm and a pipeline process to
an existing VIAME install, with nothing compiled. There are two ways to add
python: an algorithm implements one interface (``example_filter.py``, an
``ImageFilter``) and is selected by name wherever that interface is
configured; a process (``example_filter_process.py``) declares its own ports
and configuration and is placed in a pipeline directly. The first is shorter;
the second allows more control of inputs and outputs.

``__init__.py`` declares both, so VIAME knows their names without importing
them.

Loading it
----------

VIAME loads a python package named in ``VIAME_PYTHON_PLUGINS``, a
``:``-separated list of package names, from anywhere on the python path.
Either install the package into the VIAME install's site-packages::

    cmake -S . -B build -DVIAME_DIR=[viame-install] && cmake --install build

or leave it where it is and put its parent directory on ``PYTHONPATH``. Then::

    source [viame-install]/setup_viame.sh
    export VIAME_PYTHON_PLUGINS=example_external_plugin
    viame registry-dump --json | grep example_filter
