// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/uid.h>

#include <pybind11/pybind11.h>
#include "python_fold.h"

namespace py = pybind11;

VIAME_PYTHON_MODULE( uid, m )
{
  py::class_< viame::uid, std::shared_ptr< viame::uid > >(
    m,
    "UID" )
    .def( py::init<>() )
    .def( py::init< const std::string& >() )
    .def( py::init< const char*, size_t >() )
    .def( "is_valid", &viame::uid::is_valid )
    .def( "value", &viame::uid::value )
    .def( "size", &viame::uid::size )
    .def( "__len__", &viame::uid::size )
    .def( "__eq__", &viame::uid::operator== )
    .def( "__ne__", &viame::uid::operator!= )
    .def( "__lt__", &viame::uid::operator< )
  ;
}
