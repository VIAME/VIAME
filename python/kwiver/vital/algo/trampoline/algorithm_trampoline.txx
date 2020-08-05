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
 * \file algorithm_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of vital::algorithm
 */

#ifndef ALGORITHM_TRAMPOLINE_TXX
#define ALGORITHM_TRAMPOLINE_TXX

#include <vital/algo/algorithm.h>
#include <vital/config/config_block.h>
#include <python/kwiver/vital/util/pybind11.h>

template <class algorithm_base=kwiver::vital::algorithm>
class algorithm_trampoline : public algorithm_base
{
  public:
    using algorithm_base::algorithm_base;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        std::string,
        algorithm_base,
        type_name,
      );
    }

    kwiver::vital::config_block_sptr get_configuration() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        kwiver::vital::config_block_sptr,
        algorithm_base,
        get_configuration,
      );
    }

    void set_configuration(kwiver::vital::config_block_sptr config) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        algorithm_base,
        set_configuration,
        config
      );
    }

    bool check_configuration(kwiver::vital::config_block_sptr config) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        algorithm_base,
        check_configuration,
        config
      );
    }
};
#endif
