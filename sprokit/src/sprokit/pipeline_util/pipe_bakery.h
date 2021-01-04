// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_UTIL_PIPE_BAKERY_H
#define SPROKIT_PIPELINE_UTIL_PIPE_BAKERY_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include "path.h"
#include "pipe_declaration_types.h"

#include "cluster_info.h"

#include <vital/vital_types.h>
#include <sprokit/pipeline/types.h>

#include <iosfwd>

/**
 * \file pipe_bakery.h
 *
 * \brief Functions to bake a pipeline.
 */

namespace sprokit
{

/**
 * \brief Extract a configuration from a collection of blocks.
 *
 * \param blocks The blocks to use for baking the pipeline.
 *
 * \returns A pipeline baked from \p blocks.
 */
SPROKIT_PIPELINE_UTIL_EXPORT pipeline_t bake_pipe_blocks(pipe_blocks const& blocks);

/**
 * \brief Extract a configuration from a collection of blocks.
 *
 * \param blocks The blocks to use for baking the cluster.
 *
 * \returns Information about the cluster based on \p blocks.
 */
SPROKIT_PIPELINE_UTIL_EXPORT cluster_info_t bake_cluster_blocks(cluster_blocks const& blocks);

/**
 * \brief Extract a configuration from a collection of blocks.
 *
 * \param blocks The blocks to use for baking the pipeline.
 *
 * \returns A configuration extracted from \p blocks.
 */
SPROKIT_PIPELINE_UTIL_EXPORT kwiver::vital::config_block_sptr
  extract_configuration(pipe_blocks const& blocks);

}

#endif // SPROKIT_PIPELINE_UTIL_PIPE_BAKERY_H
