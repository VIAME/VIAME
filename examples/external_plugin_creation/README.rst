
========================
External Plugin Creation
========================

This document corresponds to `this example online`_, in addition to the
examples/external_plugin_creation folder in a VIAME installation.

.. _this example online: https://github.com/Kitware/VIAME/tree/master/examples/external_plugin_creation

This directory contains the source files needed to make a loadable
algorithm plugin implementation external to VIAME, which links
against an installation, or in the case of python generates a loadable script.
This is for cases where we might want to just make a plugin against pre-compiled binaries
but not build all of VIAME itself.
