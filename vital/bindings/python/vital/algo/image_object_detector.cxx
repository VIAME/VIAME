/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

/**
 * \file image_object_detector.cxx
 *
 * \brief Python bindings for \link vital::algo::image_object_detector\endlink
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vital/algo/image_object_detector.h>
#include <vital/algo/trampoline/algorithm_trampoline.tcc>
#include <vital/algo/trampoline/image_object_detector_trampoline.tcc>

using namespace kwiver::vital::algo;
using namespace kwiver::vital;
namespace py = pybind11;

PYBIND11_MODULE(image_object_detector, m)
{
  py::class_<algorithm, std::shared_ptr<algorithm>, py_algorithm<>>(m, "_algorithm")
    .def("get_configuration", &algorithm::get_configuration)
    .def("set_configuration", &algorithm::set_configuration)
    .def("check_configuration", &algorithm::check_configuration);
  
  py::class_< algorithm_def<image_object_detector>, 
              algorithm,
              std::shared_ptr<algorithm_def<image_object_detector>>, 
              py_iod_algorithm_def<>>(m, "_algorithm<image_object_detector>")
    .def(py::init())
    .def_static("create", &algorithm_def<image_object_detector>::create)
    .def_static("registered_names", &algorithm_def<image_object_detector>::registered_names)
    .def_static("get_nested_algo_configuration", &algorithm_def<image_object_detector>::get_nested_algo_configuration)
    .def_static("set_nested_algo_configuration", &algorithm_def<image_object_detector>::set_nested_algo_configuration)
    .def_static("check_nested_algo_configuration", &algorithm_def<image_object_detector>::check_nested_algo_configuration);


  py::class_<image_object_detector, std::shared_ptr<image_object_detector>, 
            algorithm_def<image_object_detector>,
              py_image_object_detector<>>(m, "ImageObjectDetector")
    .def(py::init())
    .def_static("static_type_name", &image_object_detector::static_type_name)
    .def("detect", &image_object_detector::detect);
}
