// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C++ Helper utilities for C interface of vital::config_block
 *
 * Private header for use in cxx implementation files.
 */

#ifndef VITAL_C_HELPERS_CONFIG_BLOCK_H_
#define VITAL_C_HELPERS_CONFIG_BLOCK_H_

#include <vital/config/config_block.h>

#include <vital/bindings/c/config_block.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

  extern SharedPointerCache< kwiver::vital::config_block,
                           vital_config_block_t > CONFIG_BLOCK_SPTR_CACHE;

} }

#endif //VITAL_C_HELPERS_CONFIG_BLOCK_H_
