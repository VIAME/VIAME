/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "orphan_process.h"

namespace vistk
{

orphan_process
::orphan_process(config_t const& config)
  : process(config)
{
}

orphan_process
::~orphan_process()
{
}

process_registry::type_t
orphan_process
::type() const
{
  return process_registry::type_t("orphan_process");
}

}
