/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTIL_PIPE_BAKERY_H
#define VISTK_PIPELINE_UTIL_PIPE_BAKERY_H

#include "pipeline_util-config.h"

#include "pipe_declaration_types.h"

#include <vistk/pipeline/types.h>

#include <boost/filesystem/path.hpp>

#include <istream>

/**
 * \file pipe_bakery.h
 *
 * \brief Functions to bake a pipeline.
 */

namespace vistk
{

/**
 * \brief Convert a pipeline description file into a pipeline.
 *
 * \param fname The file to load the pipeline from.
 *
 * \returns A new pipeline baked from the given file.
 */
pipeline_t VISTK_PIPELINE_UTIL_EXPORT bake_pipe_from_file(boost::filesystem::path const& fname);

/**
 * \brief Bake a pipeline from a stream.
 *
 * \param istr The stream to load the pipeline from.
 * \param inc_root The root directory to search for includes from.
 *
 * \returns A pipeline baked from the given stream.
 */
pipeline_t VISTK_PIPELINE_UTIL_EXPORT bake_pipe(std::istream& istr, boost::filesystem::path const& inc_root = "");

/**
 * \brief Extracts a configuration from a collection of blocks.
 *
 * \param blocks The blocks to use for baking the pipeline.
 *
 * \returns A pipeline baked from \p blocks.
 */
pipeline_t VISTK_PIPELINE_UTIL_EXPORT bake_pipe_blocks(pipe_blocks const& blocks);

/**
 * \brief Extracts a configuration from a collection of blocks.
 *
 * \param blocks The blocks to use for baking the pipeline.
 *
 * \returns A configuration extracted from \p blocks.
 */
config_t VISTK_PIPELINE_UTIL_EXPORT extract_configuration(pipe_blocks const& blocks);

}

#endif // VISTK_PIPELINE_UTIL_PIPE_BAKERY_H
