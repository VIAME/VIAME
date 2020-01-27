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
 * \file transform_2d_io_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<transform_2d_io> and transform_2d_io
 */

#ifndef TRANSFORM_2D_IO_TRAMPOLINE_TXX
#define TRANSFORM_2D_IO_TRAMPOLINE_TXX


#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/transform_2d_io.h>


template < class algorithm_def_t2dio_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::transform_2d_io > >
class algorithm_def_t2dio_trampoline :
      public algorithm_trampoline<algorithm_def_t2dio_base>
{
  public:
    using algorithm_trampoline<algorithm_def_t2dio_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::transform_2d_io>,
        type_name,
      );
    }
};


template< class transform_2d_io_base=
                kwiver::vital::algo::transform_2d_io >
class transform_2d_io_trampoline :
      public algorithm_def_t2dio_trampoline< transform_2d_io_base >
{
  public:
    using algorithm_def_t2dio_trampoline< transform_2d_io_base>::
              algorithm_def_t2dio_trampoline;

    kwiver::vital::transform_2d_sptr
    load_( std::string const& filename )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::transform_2d_sptr,
        kwiver::vital::algo::transform_2d_io,
        load_,
        filename
      );
    }

    void
    save_( std::string const& filename,
           kwiver::vital::transform_2d_sptr data ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::transform_2d_io,
        save_,
        filename,
        data
      );
    }
};

#endif
