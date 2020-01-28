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
 * \file detected_object_set_output_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of algorithm_def<detected_object_set_output> and detected_object_set_output
 */

#ifndef DETECTED_OBJECT_SET_OUTPUT_TRAMPOLINE_TXX
#define DETECTED_OBJECT_SET_OUTPUT_TRAMPOLINE_TXX


#include <vital/bindings/python/vital/util/pybind11.h>
#include <vital/algo/detected_object_set_output.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/image_container.h>
#include <vital/bindings/python/vital/algo/trampoline/algorithm_trampoline.txx>



template <class algorithm_def_doso_base=kwiver::vital::algorithm_def<kwiver::vital::algo::detected_object_set_output>>
class algorithm_def_doso_trampoline :
      public algorithm_trampoline<algorithm_def_doso_base>
{
  public:
    using algorithm_trampoline<algorithm_def_doso_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::detected_object_set_output>,
        type_name,
      );
    }
};


template <class detected_object_set_output_base=kwiver::vital::algo::detected_object_set_output>
class detected_object_set_output_trampoline :
      public algorithm_def_doso_trampoline<detected_object_set_output_base>
{
  public:
    using algorithm_def_doso_trampoline<detected_object_set_output_base>::
              algorithm_def_doso_trampoline;

    void write_set(kwiver::vital::detected_object_set_sptr set, std::string const& image_path) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::detected_object_set_output,
        write_set,
        set,
	image_path
      );
    }

    void open(std::string const& filename) override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::detected_object_set_output,
        open,
	filename
      );
    }

    void close() override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::detected_object_set_output,
        close,
      );
    }

    void complete() override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
	kwiver::vital::algo::detected_object_set_output,
	complete,
      );
    }
};

#endif
