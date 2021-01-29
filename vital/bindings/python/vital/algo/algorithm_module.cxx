/*ckwg +29
 * Copyright 2019-2020 by Kitware, Inc.
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
 * \file algorithm_implementation.cxx
 *
 * \brief python bindings for algorithm
 */


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include <vital/algo/algorithm.h>

#include <vital/algo/image_object_detector.h>
#include <vital/algo/refine_detections.h>
#include <vital/algo/train_detector.h>

#include <vital/bindings/python/vital/algo/algorithm.h>
#include <vital/bindings/python/vital/algo/detected_object_set_input.h>
#include <vital/bindings/python/vital/algo/detected_object_set_output.h>
#include <vital/bindings/python/vital/algo/image_filter.h>
#include <vital/bindings/python/vital/algo/image_object_detector.h>
#include <vital/bindings/python/vital/algo/refine_detections.h>
#include <vital/bindings/python/vital/algo/train_detector.h>

#include <vital/bindings/python/vital/algo/trampoline/detected_object_set_input_trampoline.txx>
#include <vital/bindings/python/vital/algo/trampoline/detected_object_set_output_trampoline.txx>
#include <vital/bindings/python/vital/algo/trampoline/image_filter_trampoline.txx>
#include <vital/bindings/python/vital/algo/trampoline/train_detector_trampoline.txx>
#include <vital/bindings/python/vital/algo/trampoline/image_object_detector_trampoline.txx>
#include <vital/bindings/python/vital/algo/trampoline/refine_detections_trampoline.txx>

#include <sstream>

namespace py = pybind11;

PYBIND11_MODULE(algorithm, m)
{
  algorithm(m);

  register_algorithm<kwiver::vital::algo::detected_object_set_input,
	    algorithm_def_dosi_trampoline<>>(m, "detected_object_set_input");
  detected_object_set_input(m);

  register_algorithm<kwiver::vital::algo::detected_object_set_output,
	    algorithm_def_doso_trampoline<>>(m, "detected_object_set_output");
  detected_object_set_output(m);

  register_algorithm<kwiver::vital::algo::image_filter,
    algorithm_def_if_trampoline<>>(m, "image_filter");
  image_filter(m);

  register_algorithm<kwiver::vital::algo::image_object_detector,
    algorithm_def_iod_trampoline<>>(m, "image_object_detector");
  image_object_detector(m);

  register_algorithm<kwiver::vital::algo::refine_detections,
    algorithm_def_rd_trampoline<>>(m, "refine_detections");
  refine_detections(m);

  register_algorithm<kwiver::vital::algo::train_detector,
    algorithm_def_td_trampoline<>>(m, "train_detector");
  train_detector(m);

}
