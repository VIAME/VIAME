This directory contains the source files needed to make a loadable
detector algorithm implementation in Python. The intent is to copy the
contents of this directory to a new folder in the plugins directory or
and existing folder.

Note: a package that ships with VIAME is named in BUILTIN_PLUGIN_PACKAGES
in kwiver/vital/plugins/discovery.py. One that does not ship with VIAME is
named by the user in VIAME_PYTHON_PLUGINS, a colon-separated list of package
names; nothing has to be added to setup_viame.sh for either.



Change the following place holders to instantiate a new detector.

@template@ - name of the detector.

@template_dir@ - name of the source subdirectory containing the detector
files. For example if the detector is in the directory plugins/fin_fish_detector,
then 'template_dir' should be replaced with 'fin_fish_detector'. This directory
can contain multiple detectors and/or filters.

The place holders also appear in capital letters indicating that the
replacement string should be capitalized.
