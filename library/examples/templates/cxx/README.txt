This directory contains the source files needed to make a loadable
detector algorithm implementation in C++. The intent is to copy the
contents of this directory to a new directory under library/, named for
what the detector does, and to add that directory to
library/CMakeLists.txt with add_subdirectory().

The detector declares its name, description and configuration in
PLUGGABLE_IMPL, and register_algorithms.cxx registers it with
register_algorithm<>() from
<viame/algorithm_framework/plugin/register_algorithm.h>.

Change the following place holders to instantiate a new detector.

@template@ - name of the detector.

@template_lib@ - name of the library that will contain the detector. Can
be the same name as the detector.

@template_dir@ - name of the source subdirectory containing the detector
files. For example if the detector is in the directory
library/fin_fish_detector, then 'template_dir' should be replaced
with 'fin_fish_detector'.

The place holders also appear in capital letters indicating that the
replacement string should be capitalized.
