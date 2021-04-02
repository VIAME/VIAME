// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/metadata_map.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
namespace py = pybind11;
namespace kv = kwiver::vital;

PYBIND11_MODULE(type_check, m)
{
    m.def("get_uint64_rep", []()
    {
        return kv::demangle(typeid(uint64_t).name());
    });
}
