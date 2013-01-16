/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTIL_PIPE_BAKERY_H
#define VISTK_PIPELINE_UTIL_PIPE_BAKERY_H

#include "pipeline_util-config.h"

#include "pipe_declaration_types.h"

#include <vistk/utilities/path.h>

#include <vistk/pipeline/types.h>

#include <iosfwd>

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
pipeline_t VISTK_PIPELINE_UTIL_EXPORT bake_pipe_from_file(path_t const& fname);

/**
 * \brief Bake a pipeline from a stream.
 *
 * \param istr The stream to load the pipeline from.
 * \param inc_root The root directory to search for includes from.
 *
 * \returns A pipeline baked from the given stream.
 */
pipeline_t VISTK_PIPELINE_UTIL_EXPORT bake_pipe(std::istream& istr, path_t const& inc_root = "");

/**
 * \brief Extract a configuration from a collection of blocks.
 *
 * \param blocks The blocks to use for baking the pipeline.
 *
 * \returns A pipeline baked from \p blocks.
 */
pipeline_t VISTK_PIPELINE_UTIL_EXPORT bake_pipe_blocks(pipe_blocks const& blocks);

/**
 * \class cluster_info pipe_bakery.h <vistk/pipeline_util/pipe_bakery.h>
 *
 * \brief Information about a loaded cluster.
 */
class VISTK_PIPELINE_UTIL_EXPORT cluster_info
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type_ The type of the cluster.
     * \param description_ A description of the cluster.
     * \param ctor_ A function to create an instance of the cluster.
     */
    cluster_info(process::type_t const& type_,
                 process_registry::description_t const& description_,
                 process_ctor_t const& ctor_);
    /**
     * \brief Destructor.
     */
    ~cluster_info();

    /// The type of the cluster.
    process::type_t const type;
    /// A description of the cluster.
    process_registry::description_t const description;
    /// A function to create an instance of the cluster.
    process_ctor_t const ctor;
};
/// A handle to information about a cluster.
typedef boost::shared_ptr<cluster_info> cluster_info_t;

/**
 * \brief Convert a cluster description file into a cluster.
 *
 * \param fname The file to load the cluster from.
 *
 * \returns Information about the cluster in the file.
 */
cluster_info_t VISTK_PIPELINE_UTIL_EXPORT bake_cluster_from_file(path_t const& fname);

/**
 * \brief Bake a cluster from a stream.
 *
 * \param istr The stream to load the cluster from.
 * \param inc_root The root directory to search for includes from.
 *
 * \returns Information about the cluster in the stream.
 */
cluster_info_t VISTK_PIPELINE_UTIL_EXPORT bake_cluster(std::istream& istr, path_t const& inc_root = "");

/**
 * \brief Extract a configuration from a collection of blocks.
 *
 * \param blocks The blocks to use for baking the cluster.
 *
 * \returns Information about the cluster based on \p blocks.
 */
cluster_info_t VISTK_PIPELINE_UTIL_EXPORT bake_cluster_blocks(cluster_blocks const& blocks);

/**
 * \brief Extract a configuration from a collection of blocks.
 *
 * \param blocks The blocks to use for baking the pipeline.
 *
 * \returns A configuration extracted from \p blocks.
 */
config_t VISTK_PIPELINE_UTIL_EXPORT extract_configuration(pipe_blocks const& blocks);

}

#endif // VISTK_PIPELINE_UTIL_PIPE_BAKERY_H
