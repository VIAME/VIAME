kwiver
======

The python bindings for the code VIAME imported from kwiver: the types,
configuration, plugin management and algorithm interfaces of
``library/core_types`` and ``library/algorithm_framework``, and the pipeline
API of ``library/pipeline_framework``.

The package is still called ``kwiver`` because everything that imports it
still spells it that way -- VIAME's own python implementations, the process
API, the pipelines. Phase 11 of the lite plan is what renames it; until then
the binding sources keep their package tree here rather than sitting beside
the C++ they bind, because forty-eight of them share a name with it.

``VERSION.txt`` is kwiver's version, not VIAME's, for the same reason.
