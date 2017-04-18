This directory contains the source files needed to make a loadable
detector algorithm implementation. The intent is to copy the contents
of this directory to a new folder in the plugins directory.



Change the following place holders to instantiate a new detector.

@template@ - name of the detector.

@template_dir@ - name of the source subdirectory containing the detector
files. For example if the detector is in the directory plugins/fin_fish_detector,
then 'template_dir' should be replaced with 'fin_fish_detector'. This directory
can contain multiple detectors and/or filters.

The place holders also appear in capital letters indicating that the
replacement string should be capitalized.
