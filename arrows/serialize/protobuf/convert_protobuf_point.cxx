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


#include "convert_protobuf_point.h"

#include <vital/exceptions.h>

#include <vital/types/protobuf/covariance.pb.h>
#include <vital/types/protobuf/point.pb.h>

#include <sstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ============================================================================
// ---- point converter template
  template < class PROTO_POINT, class POINT, int N >
void
convert_from_protobuf( const PROTO_POINT& proto_point,
                  POINT& point )
{
  if (proto_point.value_size() != N )
  {
    std::stringstream msg;
    msg << "Expecting " << N <<" values from protobuf point";
    VITAL_THROW( vital::serialization_exception, msg.str() );
  }

  typename POINT::vector_type values;
  for( int i = 0; i < N; ++i )
  {
    values[i] = proto_point.value(i);
  }
  point.set_value( values );

  typename POINT::covariance_type cov;
  convert_protobuf( proto_point.cov(), cov );

  point.set_covariance( cov );
}

// ----------------------------------------------------------------------------
  template < class POINT, class PROTO_POINT, int N >
void
convert_to_protobuf( const POINT& point, PROTO_POINT& proto_point )
{
  auto values = point.value();

  if (values.size() != N )
  {
    std::stringstream msg;
    msg << "Expecting " << N <<" values from vital; point";
    VITAL_THROW( ::kwiver::vital::serialization_exception, msg.str() );
  }

  for (int i = 0; i < N; ++i)
  {
    proto_point.add_value( values[i] );
  }

  // Covariance
  auto proto_cov = proto_point.mutable_cov();
  auto cov = point.covariance();
  convert_protobuf( cov, *proto_cov );
}

// ----------------------------------------------------------------------------
#define CONVERT( PT, VT, C )                                    \
void convert_protobuf( const PT& proto_point, VT& point )       \
{  convert_from_protobuf<PT, VT, C>( proto_point, point ); }    \
void convert_protobuf( const VT& point, PT& proto_point )       \
{ convert_to_protobuf<VT, PT, C> (point, proto_point ); }

CONVERT( ::kwiver::protobuf::point_i, ::kwiver::vital::point_2i, 2 )
CONVERT( ::kwiver::protobuf::point_d, ::kwiver::vital::point_2d, 2 )
CONVERT( ::kwiver::protobuf::point_d, ::kwiver::vital::point_2f, 2 )
CONVERT( ::kwiver::protobuf::point_d, ::kwiver::vital::point_3d, 3 )
CONVERT( ::kwiver::protobuf::point_d, ::kwiver::vital::point_3f, 3 )
CONVERT( ::kwiver::protobuf::point_d, ::kwiver::vital::point_4d, 4 )
CONVERT( ::kwiver::protobuf::point_d, ::kwiver::vital::point_4f, 4 )

#undef CONVERT


// ============================================================================
// -- covariance converter template
template <class COV>
void
convert_from_protobuf( const ::kwiver::protobuf::covariance& proto_covariance,
                       COV& covariance )
{
  typename COV::data_type values[COV::data_size];
  for ( unsigned i = 0; i < COV::data_size; ++i )
  {
    values[i] = proto_covariance.value(i);
  }
  covariance.set_data( values );
}


template <class COV>
void
convert_to_protobuf( const COV& covariance,
                     ::kwiver::protobuf::covariance& proto_covariance )
{
  const auto* data = covariance.data();

  proto_covariance.set_dim( COV::data_size );

  for (unsigned i = 0; i < COV::data_size; ++i )
  {
    proto_covariance.add_value( data[i] );
  }
}

// ============================================================================
#define CONVERT( VT )                                                   \
void convert_protobuf( const ::kwiver::protobuf::covariance& proto_covariance, \
                       VT& covariance )                                 \
{ convert_from_protobuf<VT>( proto_covariance, covariance ); }          \
void convert_protobuf( const VT& covariance,                            \
                       ::kwiver::protobuf::covariance& proto_covariance ) \
{ convert_to_protobuf<VT>(covariance, proto_covariance ); }

CONVERT( ::kwiver::vital::covariance_2d )
CONVERT( ::kwiver::vital::covariance_2f )
CONVERT( ::kwiver::vital::covariance_3d )
CONVERT( ::kwiver::vital::covariance_3f )
CONVERT( ::kwiver::vital::covariance_4d )
CONVERT( ::kwiver::vital::covariance_4f )

#undef CONVERT
} } } } // end namespace
