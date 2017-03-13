Build Tips 'n Tricks
--------------------

* Super-Build Optimization:

When VIAME is build as a super-build multiple solutions or makefiles are generated
for each individual project in the super-build. These can be opened up if you want
to experiment with changes in one and not rebuild the entire superbuild. VIAME
places these projects in [build-directory]/build/src/* and fletch in
[build-directory]/build/src/fletch-build/build/src/*. You can also run ccmake or
the cmake GUI in these locations, which can let you manually change the build settings
for sub-projects (say, for example, if one doesn't build).


* Python:

The default Python used is 2.7, though other versions may work as well. It depends on
your build settings and which dependency projects are turned on (some require 2.7, some
don't).


