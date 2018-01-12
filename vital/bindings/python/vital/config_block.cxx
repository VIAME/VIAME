/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vital/config/config_block.h>
#include <vital/config/config_block_io.h>

namespace py = pybind11;

typedef kwiver::vital::config_block config_block;

// A lot of config block functions can be used as is by pybind11
// but we don't want pybind11 to deal with templates, so we'll make
// some simple pass-through helper functions

void
set_value(config_block &self, std::string key, py::object value_obj)
{
  // We convert to a string regardless of input type. The process can reinterpret as needed.
  std::string value = py::str(value_obj).cast<std::string>();
  return self.set_value<std::string>(key, value);
}

void
set_value2(config_block &self, std::string key, py::object value_obj, std::string desc)
{
  // We convert to a string regardless of input type. The process can reinterpret as needed.
  std::string value = py::str(value_obj).cast<std::string>();
  return self.set_value<std::string>(key, value, desc);
}

std::string
get_value(config_block &self, std::string key)
{
  return self.get_value<std::string>(key);
}

std::string
get_value2(config_block &self, std::string key, std::string def)
{
  return self.get_value<std::string>(key, def);
}

bool
get_value_bool(config_block &self, std::string key)
{
  return self.get_value<bool>(key);
}

bool
get_value_bool2(config_block &self, std::string key, bool def)
{
  return self.get_value<bool>(key, def);
}

// It doesn't like this one without a pass-through, either
std::shared_ptr<config_block>
read_config_file(std::string filename, std::string application_name,
                 std::string application_version, std::string install_prefix,
                 bool merge)
{
  return kwiver::vital::read_config_file(filename, application_name,
                                         application_version, install_prefix, merge);
}

PYBIND11_MODULE(config_block, m)
{
  py::class_<config_block, std::shared_ptr<config_block>>(m, "ConfigBlock")
  .def(py::init(&config_block::empty_config),
    py::arg("name")=std::string())
  .def_static("from_file", &read_config_file,
    py::arg("filename"), py::arg("application_name")="", py::arg("application_version")="",
    py::arg("install_prefix")="", py::arg("merge")=false)
  .def("write", &kwiver::vital::write_config_file,
    py::arg("filename"))
  .def("set_value", &set_value,
    py::arg("key"), py::arg("value"))
  .def("set_value", &set_value2,
    py::arg("key"), py::arg("value"), py::arg("desc"))
  .def("unset_value", &config_block::unset_value,
    py::arg("key"))
  .def("get_value", &get_value,
    py::arg("key"))
  .def("get_value", &get_value2,
    py::arg("key"), py::arg("default"))
  .def("get_value_bool", &get_value_bool,
    py::arg("key"))
  .def("get_value_bool", &get_value_bool2,
    py::arg("key"), py::arg("default"))
  .def("has_value", &config_block::has_value,
    py::arg("key"))
  .def("get_description", &config_block::get_description,
    py::arg("key"))
  .def("mark_read_only", &config_block::mark_read_only,
    py::arg("key"))
  .def("is_read_only", &config_block::is_read_only,
    py::arg("key"))
  .def("available_keys", &config_block::available_values)
  .def("subblock", &config_block::subblock,
    py::arg("key"))
  .def("subblock_view", &config_block::subblock_view,
    py::arg("key"))
  .def("merge_config", &config_block::merge_config,
    py::arg("config"))
  .def_property_readonly("name", &config_block::get_name)
  .def_readonly_static("BLOCK_SEP", &config_block::block_sep)
  ;
}
