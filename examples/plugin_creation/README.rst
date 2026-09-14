
===============
Plugin Creation
===============

This document corresponds to the `Plugin Creation`_ example in a VIAME desktop installation.

.. _Plugin Creation: https://github.com/VIAME/VIAME/tree/master/examples/plugin_creation

------------------------
External Plugin Creation
------------------------

This directory contains the source files needed to make a loadable algorithm plugin implementation
external to VIAME, which links against an installation, or in the case of python generates a loadable
script. This is for cases where we might want to just make a plugin against pre-compiled binaries,
instead of building all of VIAME itself.

The procedure is slightly different depending on whether you are developing an external C++ or
Python module. C++ modules require linking your module against VIAME, the output of which is 
plugin DLL which can be used directly in VIAME pipelines. Python processes can be made without
compilation, and placed in your PYTHONPATH for use by the plugin system.

------------------------
Internal Plugin Creation
------------------------

An algorithm or process built with VIAME itself goes in a directory under
``library/``, named for what it does, beside the libraries it belongs with.
``library/examples/templates`` holds a C++ and a python detector to start from,
with ``@template@`` placeholders; each has a README saying what to replace.
``library/examples`` holds the hello world detector and filter, built and
registered, with ``README_algorithms.txt`` walking through them, and
``library/examples/README.rst`` covers the sprokit process template.

A C++ implementation declares its configuration with ``PLUGGABLE_IMPL`` and
is registered in its library's ``register_algorithms.cxx`` with
``register_algorithm<>()``; a python one is named in its package's
``__init__.py`` declarations. Neither needs anything loaded at run time: VIAME
registers what it was built with directly.
