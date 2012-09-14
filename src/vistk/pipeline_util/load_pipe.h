/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTIL_LOAD_PIPE_H
#define VISTK_PIPELINE_UTIL_LOAD_PIPE_H

#include "pipeline_util-config.h"

#include "pipe_declaration_types.h"

#include <vistk/utilities/path.h>

#include <iosfwd>

/**
 * \file load_pipe.h
 *
 * \brief Load a pipeline declaration from a stream.
 */

namespace vistk
{

/**
 * \brief Convert a pipeline description file into a collection of pipeline blocks.
 *
 * \param fname The file to load the pipeline blocks from.
 *
 * \returns A new set of pipeline blocks.
 */
VISTK_PIPELINE_UTIL_EXPORT pipe_blocks load_pipe_blocks_from_file(path_t const& fname);

/**
 * \brief Convert a pipeline description into a pipeline.
 *
 * \param istr The stream to load the pipeline from.
 * \param inc_root The root directory to search for includes from.
 *
 * \returns A new set of pipeline blocks.
 */
VISTK_PIPELINE_UTIL_EXPORT pipe_blocks load_pipe_blocks(std::istream& istr, path_t const& inc_root = "");

/**
 * \brief Convert a cluster description file into a collection of cluster blocks.
 *
 * \param fname The file to load the cluster blocks from.
 *
 * \returns A new set of cluster blocks.
 */
cluster_blocks VISTK_PIPELINE_UTIL_EXPORT load_cluster_blocks_from_file(path_t const& fname);

/**
 * \brief Convert a cluster description into a cluster.
 *
 * \param istr The stream to load the cluster from.
 * \param inc_root The root directory to search for includes from.
 *
 * \returns A new set of cluster blocks.
 */
cluster_blocks VISTK_PIPELINE_UTIL_EXPORT load_cluster_blocks(std::istream& istr, path_t const& inc_root = "");

}

#endif // VISTK_PIPELINE_UTIL_LOAD_PIPE_H
