// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "orphan_process.h"

/**
 * \file orphan_process.cxx
 *
 * \brief Implementation of the orphan process.
 */

namespace sprokit
{

orphan_process
::orphan_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
{
}

orphan_process
::~orphan_process()
{
}

}
