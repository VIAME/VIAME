This directory contains the source files needed to make a loadable
detector algorithm implementation in Python. The intent is to copy the
contents of this directory to a new directory, or into an existing
package.

__init__.py declares what the package provides: one line per algorithm,
naming its interface, its registered name, a description and the
"module:Class" it lives in. Nothing is imported until a pipeline asks for
the algorithm, so a detector that imports torch costs nothing at startup.

A package that ships with VIAME is named in BUILTIN_PLUGIN_PACKAGES in
the viame.plugins.discovery module. One that does not ship with VIAME is
named by the user in VIAME_PYTHON_PLUGINS, a colon-separated list of
package names; nothing has to be added to setup_viame.sh for either.

Change the following place holders to instantiate a new detector.

@template@ - name of the detector.

@template_dir@ - name of the package containing the detector files. For
example if the detector is in the directory library/fin_fish_detector,
then 'template_dir' should be replaced with 'fin_fish_detector'. This
package can contain multiple detectors and/or filters, each with a line
in __init__.py.

The place holders also appear in capital letters indicating that the
replacement string should be capitalized.
