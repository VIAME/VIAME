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

/**
 * \file read_track_descriptor_set_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<read_track_descriptor_set> and read_track_descriptor_set
 */

#ifndef READ_TRACK_DESCRIPTOR_SET_TRAMPOLINE_TXX
#define READ_TRACK_DESCRIPTOR_SET_TRAMPOLINE_TXX


#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/read_track_descriptor_set.h>


template < class algorithm_def_rtds_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::read_track_descriptor_set > >
class algorithm_def_rtds_trampoline :
      public algorithm_trampoline<algorithm_def_rtds_base>
{
  public:
    using algorithm_trampoline<algorithm_def_rtds_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::read_track_descriptor_set>,
        type_name,
      );
    }
};


template< class read_track_descriptor_set_base=
                kwiver::vital::algo::read_track_descriptor_set >
class read_track_descriptor_set_trampoline :
      public algorithm_def_rtds_trampoline< read_track_descriptor_set_base >
{
  public:
    using algorithm_def_rtds_trampoline< read_track_descriptor_set_base>::
              algorithm_def_rtds_trampoline;
    void
    open( std::string const& filename ) override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::read_track_descriptor_set,
        open,
        filename
      );
    }

    void
    close() override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::read_track_descriptor_set,
        close,
      );
    }

    bool
    read_set( kwiver::vital::track_descriptor_set_sptr& descriptor_set ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algo::read_track_descriptor_set,
        read_set,
        descriptor_set
      );
    }
};

#endif
