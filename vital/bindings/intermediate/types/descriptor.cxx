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

#include <vital/bindings/c/types/descriptor.h>

#include <pybind11/stl.h>

#include <list>

namespace py = pybind11;

PYBIND11_MODULE(_descriptor, m)
{
  py::class_<vital_descriptor_s>(m, "vital_descriptor_s");

  m.def("vital_descriptor_destroy", &vital_descriptor_destroy
    , py::arg("size"), py::arg("eh"));
  m.def("vital_descriptor_size", &vital_descriptor_size
    , py::arg("size"), py::arg("eh"));
  m.def("vital_descriptor_as_bytes"
    , [](vital_descriptor_t *d, vital_error_handle_t *eh)
      {
        size_t size = vital_descriptor_size(d, eh);
        unsigned char *uc_array = vital_descriptor_as_bytes(d,eh);
        std::string s(reinterpret_cast<char const*>(uc_array), size);
        return py::bytes(s);
      }
    );
  m.def("vital_descriptor_type_name", &vital_descriptor_type_name
    , py::arg("size"), py::arg("eh")
    , py::return_value_policy::reference);
  m.def("vital_descriptor_new_d", &vital_descriptor_new_d
    , py::arg("size"), py::arg("eh")
    , py::return_value_policy::reference);
  m.def("vital_descriptor_get_d_raw_data"
    , [](vital_descriptor_t *d, vital_error_handle_t *eh)
      {
        double *tmp = vital_descriptor_get_d_raw_data(d, eh);
        size_t size = vital_descriptor_size(d, eh);
        std::list<double> ret_list;
        for(size_t i = 0; i < size; i++)
        {
          ret_list.push_back(tmp[i]);
        }
        return ret_list;
      }
    , py::return_value_policy::reference);
  m.def("vital_descriptor_new_f", &vital_descriptor_new_f
    , py::arg("size"), py::arg("eh")
    , py::return_value_policy::reference);
  m.def("vital_descriptor_get_f_raw_data"
    , [](vital_descriptor_t *d, vital_error_handle_t *eh)
      {
        float *tmp = vital_descriptor_get_f_raw_data(d, eh);
        size_t size = vital_descriptor_size(d, eh);
        std::list<float> ret_list;
        for(size_t i = 0; i < size; i++)
        {
          ret_list.push_back(tmp[i]);
        }
        return ret_list;
      }
    , py::return_value_policy::reference);
}
