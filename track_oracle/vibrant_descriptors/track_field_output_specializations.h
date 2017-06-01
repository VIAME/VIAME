/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_FIELD_VIBRANT_DESCRIPTORS_OUTPUT_SPECIALIZATIONS_H
#define INCL_TRACK_FIELD_VIBRANT_DESCRIPTORS_OUTPUT_SPECIALIZATIONS_H

#include <vital/vital_config.h>
#include <track_oracle/vibrant_descriptors/vibrant_descriptors_export.h>

#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/vibrant_descriptors/descriptor_cutic_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_metadata_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_motion_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_overlap_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_event_label_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_raw_1d_type.h>

namespace kwiver {
namespace track_oracle {

template< >
VIBRANT_DESCRIPTORS_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< descriptor_cutic_type >& f);

template< >
VIBRANT_DESCRIPTORS_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< descriptor_metadata_type >& f );

template< >
VIBRANT_DESCRIPTORS_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< descriptor_motion_type >& f );

template< >
VIBRANT_DESCRIPTORS_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< descriptor_overlap_type >& f );

template< >
VIBRANT_DESCRIPTORS_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< descriptor_event_label_type >& f );

template< >
VIBRANT_DESCRIPTORS_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< descriptor_raw_1d_type >& f );

} // ...track_oracle
} // ...kwiver

#endif
