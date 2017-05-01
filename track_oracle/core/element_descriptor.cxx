/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "element_descriptor.h"


using std::string;

namespace kwiver {
namespace track_oracle {


string
element_descriptor
::role2str( element_role e )
{
  switch (e)
  {
  case INVALID: return "invalid";
  case SYSTEM: return "system";
  case WELLKNOWN: return "well-known";
  case ADHOC: return "ad-hoc";
  default: return "unknown?";
  }
}

element_descriptor::element_role
element_descriptor
::str2role( const string& s )
{
       if (s == "system") return SYSTEM;
  else if (s == "well-known") return WELLKNOWN;
  else if (s == "ad-hoc") return ADHOC;
  else return INVALID;
}

} // ...track_oracle
} // ...kwiver
