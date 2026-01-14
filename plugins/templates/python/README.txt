This directory contains the source files needed to make a loadable
detector algorithm implementation in Python. The intent is to copy the
contents of this directory to a new folder in the plugins directory or
and existing folder.

Note: if made in a new plugins directory, the setup_viame.sh or .bat
script should also be modified to include an import for the new plugin
(e.g. SPROKIT_PYTHON_MODULES should contain viame.@template_dir@
either via adding it to setup_viame script or alternatively the environment
to load the new python plugin folder).



Change the following place holders to instantiate a new detector.

@template@ - name of the detector.

@template_dir@ - name of the source subdirectory containing the detector
files. For example if the detector is in the directory plugins/fin_fish_detector,
then 'template_dir' should be replaced with 'fin_fish_detector'. This directory
can contain multiple detectors and/or filters.

The place holders also appear in capital letters indicating that the
replacement string should be capitalized.
