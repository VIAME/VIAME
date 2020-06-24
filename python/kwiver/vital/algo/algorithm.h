/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#ifndef KWIVER_VITAL_PYTHON_ALGORITHM_H_
#define KWIVER_VITAL_PYTHON_ALGORITHM_H_

#include <pybind11/pybind11.h>
namespace py = pybind11;

void algorithm(py::module &m);

template<class implementation, class trampoline>
void register_algorithm(py::module &m,
                        const std::string implementation_name)
{
  std::stringstream impl_name;
  impl_name << "_algorithm<" << implementation_name << ">";

  py::class_< kwiver::vital::algorithm_def<implementation>,
              std::shared_ptr<kwiver::vital::algorithm_def<implementation>>,
              kwiver::vital::algorithm,
              trampoline >(m, impl_name.str().c_str())
    .def(py::init())
    .def_static("create", &kwiver::vital::algorithm_def<implementation>::create)
    .def_static("registered_names",
                &kwiver::vital::algorithm_def<implementation>::registered_names)
    .def_static("get_nested_algo_configuration",
                &kwiver::vital::algorithm_def<implementation>::get_nested_algo_configuration)
    .def_static("set_nested_algo_configuration",
                &kwiver::vital::algorithm_def<implementation>::set_nested_algo_configuration)
    .def_static("check_nested_algo_configuration",
                &kwiver::vital::algorithm_def<implementation>::check_nested_algo_configuration);
}
#endif
