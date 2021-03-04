// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the implementation for vital video metadata converters.
 */

#include "convert_metadata.h"

#include <vital/klv/klv_0601.h>
#include <vital/klv/klv_0104.h>
#include <vital/klv/klv_data.h>
#include <vital/klv/klv_parse.h>
#include <vital/exceptions/metadata.h>

namespace kwiver {
namespace vital {

const std::string convert_metadata::MISB_0104( "MISB_0104" );
const std::string convert_metadata::MISB_0601( "MISB_0601" );

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
::convert( klv_data const& klv, metadata& md )
{
  klv_uds_key uds_key( klv ); // create key from raw data

  if ( is_klv_0601_key( uds_key ) )
  {
    if ( ! klv_0601_checksum( klv ) )
    {
      // serious error
      VITAL_THROW( metadata_exception, "checksum error on 0601 packet");
    }

    klv_lds_vector_t lds = parse_klv_lds( klv );
    convert_0601_metadata( lds, md );
  }
  else if ( klv_0104::is_key( uds_key ) )
  {
    klv_uds_vector_t uds = parse_klv_uds( klv );
    convert_0104_metadata( uds,  md );
  }
  else
  {
    LOG_DEBUG( m_logger, "Unsupported UDS Key: "
              << uds_key << " data size is "
              << klv.value_size() );
  }
}

// ------------------------------------------------------------------
std::type_info const&
convert_metadata
::typeid_for_tag( vital_metadata_tag tag )
{

  switch (tag)
  {
#define VITAL_META_TRAIT_CASE(TAG, NAME, T, ...) case VITAL_META_ ## TAG: return typeid(T);

    KWIVER_VITAL_METADATA_TAGS( VITAL_META_TRAIT_CASE )

#undef VITAL_META_TRAIT_CASE

  default: return typeid(void);
  }
}

} } // end namespace
