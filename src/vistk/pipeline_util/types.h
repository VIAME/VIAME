/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTIL_TYPES_H
#define VISTK_PIPELINE_UTIL_TYPES_H

#include <boost/filesystem/path.hpp>

#include <vector>

/**
 * \file types.h
 *
 * \brief Types used in the pipeline_util library.
 */

namespace vistk
{

///
typedef boost::filesystem::path path_t;
typedef std::vector<path_t> paths_t;

}

#endif // VISTK_PIPELINE_UTIL_TYPES_H
