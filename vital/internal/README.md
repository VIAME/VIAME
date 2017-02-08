# Vital Internal Headers

This directory contains internal header that are used privately by KWIVER but
are not installed.  As such, these headers should only be included from
implementation files (.cxx) and private header files which are not installed.
Currently the internal headers include

[Cereal](https://github.com/USCiLab/cereal)
version [1.2.1](https://github.com/USCiLab/cereal/releases/tag/v1.2.1).
A header only library for serialization of data into binary, JSON, or XML formats.
