# Building the KWIVER Manual

The KWIVER manual is built using [Sphinx](sphinx-doc.org), a Python based documentation
tool.  To build the manual you will need a Python envionment with the following packages
available:

  * [Sphinx](sphinx-doc.org) The main documentation tool
  * [sphinx\_rtd\_theme](https://sphinx-rtd-theme.readthedocs.io/en/latest/)
  * [breathe](https://breathe.readthedocs.io/en/latest/) Tool to integrate [Doxygen](http://www.doxygen.nl/)
    documentation with Sphinx based documentation.

To generate the manual, in the CMake "build" directory for KWIVER enter the following commands:

`$ make doxygen-kwiver`

Then

`$ make sphinx-kwiver`

The resulting documentation will be rooted in `doc/sphinx/index.html`
