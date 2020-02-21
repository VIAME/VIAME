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
#include <python/kwiver/vital/algo/trampoline/match_descriptor_sets_trampoline.txx>
#include <python/kwiver/vital/algo/match_descriptor_sets.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void match_descriptor_sets(py::module &m)
{
  py::class_< kwiver::vital::algo::match_descriptor_sets,
              std::shared_ptr<kwiver::vital::algo::match_descriptor_sets>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::match_descriptor_sets>,
              match_descriptor_sets_trampoline<> >(m, "MatchDescriptorSets")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::match_descriptor_sets::static_type_name)
    .def("query",
        &kwiver::vital::algo::match_descriptor_sets::query)
    .def("append_to_index",
        &kwiver::vital::algo::match_descriptor_sets::append_to_index)
    .def("query_and_append",
        &kwiver::vital::algo::match_descriptor_sets::query_and_append);
}
}
}
}
