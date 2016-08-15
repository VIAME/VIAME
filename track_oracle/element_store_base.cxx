/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "element_store_base.h"

namespace kwiver {
namespace track_oracle {

const element_descriptor&
element_store_base
::get_descriptor() const
{
  return this->d;
}


element_store_base
::~element_store_base()
{}


} // ...track_oracle
} // ...kwiver
