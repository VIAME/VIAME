// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   loaded_cluster.h
 * @brief  Interface to loaded_cluster class.
 */

#ifndef SPROKIT_PIPELINE_UTIL_LOADED_CLUSTER_H
#define SPROKIT_PIPELINE_UTIL_LOADED_CLUSTER_H

#include <sprokit/pipeline/process_cluster.h>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief Friendly process cluster.
 *
 * This class provides friend access to process_cluster by
 * cluster_creator.
 *
 */
class loaded_cluster
  : public process_cluster
{
  public:
    loaded_cluster(kwiver::vital::config_block_sptr const& config);
    ~loaded_cluster();

    friend class cluster_creator;
};

} // end namespace sprokit

#endif /* SPROKIT_PIPELINE_UTIL_LOADED_CLUSTER_H */
