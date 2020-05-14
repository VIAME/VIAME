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

#include "load_save_point.h"

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/types/vector.hpp>
#include <vital/internal/cereal/types/map.hpp>
#include <vital/internal/cereal/types/utility.hpp>

#include <vital/logger/logger.h>

#include <vector>


namespace cereal {

// ============================================================================
  template <class P, unsigned N>
void save_point( ::cereal::JSONOutputArchive& archive, const P& pt )
{
  // Get values as an vector
  auto data = pt.value();
  std::vector< typename P::data_type > values;
  for ( unsigned i = 0; i < N; ++i )
  {
    values.push_back( data( i ) );
  }

  archive( CEREAL_NVP( values ) );

  // Save covariance
  save( archive, pt.covariance() );
}

// ----------------------------------------------------------------------------
  template <class P, unsigned N>
void load_point( ::cereal::JSONInputArchive& archive, P& pt )
{
  // get values as a vector
  std::vector< typename P::data_type > values;
  archive( CEREAL_NVP( values ) );

  if( values.size() != N )
  {
    LOG_ERROR( ::kwiver::vital::get_logger( "data_serializer" ),
               "Number of elements for loading a point is not as expected. "
               "Expected " << N << " but got " << values.size()
               << "." );
    return;
  }

  typename P::vector_type lpt;
  for ( unsigned i = 0; i < N; ++i )
  {
    lpt(i) = values[i];
  }

  // Construct point from vector
  pt.set_value( lpt );

  // Load covariance
  typename P::covariance_type cov;
  load( archive, cov );
  pt.set_covariance( cov );
}

// ----------------------------------------------------------------------------
#define LOAD_SAVE( P, N )                                       \
  void save( ::cereal::JSONOutputArchive& archive, const P& pt )  \
  { save_point<P, N>( archive, pt ); }                          \
  void load( ::cereal::JSONInputArchive& archive, P& pt )         \
  { load_point<P, N>(archive, pt ); }

  LOAD_SAVE( ::kwiver::vital::point_2i, 2 )
  LOAD_SAVE( ::kwiver::vital::point_2d, 2 )
  LOAD_SAVE( ::kwiver::vital::point_2f, 2 )
  LOAD_SAVE( ::kwiver::vital::point_3d, 3 )
  LOAD_SAVE( ::kwiver::vital::point_3f, 3 )
  LOAD_SAVE( ::kwiver::vital::point_4d, 4 )
  LOAD_SAVE( ::kwiver::vital::point_4f, 4 )

#undef LOAD_SAVE

// ============================================================================
template <class C>
void save_covariance( ::cereal::JSONOutputArchive& archive, const C& cov )
{
  auto* data = cov.data();
  std::vector< typename C::data_type > cov_values( data, data + C::data_size );

  archive( CEREAL_NVP( cov_values ) );
}

// ----------------------------------------------------------------------------
template <class C>
void load_covariance( ::cereal::JSONInputArchive& archive, C& cov )
{
  std::vector< typename C::data_type > cov_values;

  archive( CEREAL_NVP( cov_values ) );

  cov.set_data( cov_values.data() );

  if( cov_values.size() != C::data_size )
  {
    LOG_ERROR( ::kwiver::vital::get_logger( "data_serializer" ),
               "Number of elements for loading covariance is not as expected. "
               "Expected " << C::data_size << " but got " << cov_values.size()
               << "." );
    return;
  }
}

// ----------------------------------------------------------------------------
#define LOAD_SAVE( C )                                          \
void save( ::cereal::JSONOutputArchive& archive, const C& cov )   \
{ save_covariance<C>(archive, cov ); }                          \
void load( ::cereal::JSONInputArchive& archive, C& cov )          \
{ load_covariance<C>(archive, cov ); }

  LOAD_SAVE( ::kwiver::vital::covariance_2d )
  LOAD_SAVE( ::kwiver::vital::covariance_2f )
  LOAD_SAVE( ::kwiver::vital::covariance_3d )
  LOAD_SAVE( ::kwiver::vital::covariance_3f )
  LOAD_SAVE( ::kwiver::vital::covariance_4d )
  LOAD_SAVE( ::kwiver::vital::covariance_4f )

#undef LOAD_SAVE

} // end namespace
