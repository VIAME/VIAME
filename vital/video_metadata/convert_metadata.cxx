/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

/**
 * \file
 * \brief This file contains the implementation for vital video metadata converters.
 */

#include "convert_metadata.h"

#include <vital/klv/klv_0601.h>
#include <vital/klv/klv_0104.h>
#include <vital/klv/klv_data.h>
#include <vital/klv/klv_parse.h>
#include <vital/exceptions/klv.h>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
convert_metadata
::convert_metadata()
  : m_logger( kwiver::vital::get_logger( "vital.convert_metadata" ) )
{
  //
  // Initialize converters
  //
  convert_to_int.add_converter<uint64_t>();
  convert_to_int.add_converter<uint32_t>();
  convert_to_int.add_converter<uint16_t>();
  convert_to_int.add_converter<uint8_t>();

  convert_to_int.add_converter<int64_t>();
  convert_to_int.add_converter<int32_t>();
  convert_to_int.add_converter<int16_t>();
  convert_to_int.add_converter<int8_t>();

  convert_to_double.add_converter<double>();
  convert_to_double.add_converter<uint64_t>();
  convert_to_double.add_converter<uint32_t>();
  convert_to_double.add_converter<uint16_t>();
  convert_to_double.add_converter<uint8_t>();

  convert_to_double.add_converter<int64_t>();
  convert_to_double.add_converter<int32_t>();
  convert_to_double.add_converter<int16_t>();
  convert_to_double.add_converter<int8_t>();
}


convert_metadata
::~convert_metadata()
{  }


// ==================================================================
void convert_metadata
::convert( klv_data const& klv, video_metadata& metadata )
{
  klv_uds_key uds_key( klv ); // create key from raw data

  if ( is_klv_0601_key( uds_key ) )
  {
    if ( ! klv_0601_checksum( klv ) )
    {
      // serious error
      throw klv_exception( "checksum error on 0601 packet");
    }

    klv_lds_vector_t lds = parse_klv_lds( klv );
    convert_0601_metadata( lds, metadata );
  }
  else if ( klv_0104::is_key( uds_key ) )
  {
    klv_uds_vector_t uds = parse_klv_uds( klv );
    convert_0104_metadata( uds,  metadata );
  }
  else
  {
    LOG_DEBUG( m_logger, "Unsupported UDS Key: "
              << uds_key << " data size is "
              << klv.value_size() );
  }
}

} } // end namespace
