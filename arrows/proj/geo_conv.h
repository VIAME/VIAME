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

/**
 * \file
 * \brief PROJ geo_conversion functor interface
 */

#ifndef KWIVER_ARROWS_PROJ_GEO_CONV_H_
#define KWIVER_ARROWS_PROJ_GEO_CONV_H_


#include <vital/vital_config.h>
#include <arrows/proj/kwiver_algo_proj_export.h>

#include <vital/types/geodesy.h>

#include <unordered_map>

namespace kwiver {
namespace arrows {
namespace proj {

/// PROJ implementation of geo_conversion functor
class KWIVER_ALGO_PROJ_EXPORT geo_conversion : public vital::geo_conversion
{
public:
  geo_conversion() {}
  virtual ~geo_conversion();

  virtual char const* id() const override;

  /// Conversion operator
  virtual vital::vector_2d operator()( vital::vector_2d const& point,
                                       int from, int to ) override;

private:
  void* projection( int crs );

  std::unordered_map< int, void* > m_projections;
};

} } } // end namespace

#endif // KWIVER_ARROWS_PROJ_GEO_CONV_H_
