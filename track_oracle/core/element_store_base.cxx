// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
