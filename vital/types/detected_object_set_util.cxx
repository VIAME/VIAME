// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detected_object_set_util.h"

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
void
scale_detections( detected_object_set_sptr dos,
       double scale_factor )
{
  if( scale_factor == 1.0 )
  {
    return;
  }

  for( auto detection : *dos )
  {
    auto bbox = detection->bounding_box();
    bbox = kwiver::vital::scale( bbox, scale_factor );
    detection->set_bounding_box( bbox );
  }
}

// ------------------------------------------------------------------
void
shift_detections( detected_object_set_sptr dos,
       double col_shift, double row_shift )
{
  if( col_shift == 0.0 && row_shift == 0.0 )
  {
    return;
  }

  for( auto detection : *dos )
  {
    auto bbox = detection->bounding_box();
    bbox = kwiver::vital::translate( bbox,
      bounding_box_d::vector_type( col_shift, row_shift ) );
    detection->set_bounding_box( bbox );
  }
}

} } // end namespace
