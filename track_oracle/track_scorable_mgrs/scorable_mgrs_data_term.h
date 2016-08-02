/*ckwg +5
 * Copyright 2015-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_SCORABLE_MGRS_DATA_TERM_H
#define INCL_SCORABLE_MGRS_DATA_TERM_H

#include <vital/vital_config.h>
#include <track_oracle/track_scorable_mgrs/scorable_mgrs_data_term_export.h>

#include <track_oracle/data_terms/data_terms_common.h>
#include <track_oracle/kwiver_io_base.h>
#include <track_oracle/track_scorable_mgrs/scorable_mgrs.h>

namespace kwiver {
namespace track_oracle {

namespace dt {
namespace tracking {

DECL_DT_RW_STRXMLCSV( mgrs_pos, kwiver::track_oracle::scorable_mgrs, "MGRS location of the detection" );

} // ...tracking
} // ...dt
} // ...track_oracle
} // ...kwiver

#endif
