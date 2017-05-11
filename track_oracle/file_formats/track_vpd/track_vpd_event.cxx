/*ckwg +5
 * Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "track_vpd_event.h"

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::string;

namespace kwiver {
namespace track_oracle {

// event names chosen to be compatible with VIRAT,
// which may require some deeper thought.  We could, for example,
// call aries_interface directly.

string
track_vpd_event_type
::event_type_to_str( unsigned t )
{
  switch (t)
  {
  case 1: return "PersonLoadingVehicle";
  case 2: return "PersonUnloadingVehicle";
  case 3: return "PersonOpeningTrunk";
  case 4: return "PersonClosingTrunk";
  case 5: return "PersonEnteringVehicle";
  case 6: return "PersonExitingVehicle";
  case 7: return "PersonGesturing";
  case 8: return "PersonDigging";
  case 9: return "PersonCarrying";
  case 10: return "PersonRunning";
  case 11: return "PersonEnteringFacility";
  case 12: return "PersonExitingFacility";
  }
  return "";
}

unsigned
track_vpd_event_type
::str_to_event_type( const string& s )
{
  if (s == "PersonLoadingVehicle") return 1;
  if (s == "PersonUnloadingVehicle") return 2;
  if (s == "PersonOpeningTrunk") return 3;
  if (s == "PersonClosingTrunk") return 4;
  if (s == "PersonEnteringVehicle") return 5;
  if (s == "PersonExitingVehicle") return 6;
  if (s == "PersonGesturing") return 7;
  if (s == "PersonDigging") return 8;
  if (s == "PersonCarrying") return 9;
  if (s == "PersonRunning") return 10;
  if (s == "PersonEnteringFacility") return 11;
  if (s == "PersonExitingFacility") return 12;
  return 0;
}

} // ...track_oracle
} // ...kwiver
