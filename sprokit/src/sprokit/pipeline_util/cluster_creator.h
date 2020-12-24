// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   cluster_creator.h
 * @brief  Interface for class cluster_creator.
 */

#ifndef SPROKIT_PIPELINE_UTIL_CLUSTER_CREATOR_H
#define SPROKIT_PIPELINE_UTIL_CLUSTER_CREATOR_H

#include "cluster_bakery.h"

#include <vital/logger/logger.h>
#include <vital/config/config_block.h>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief Cluster Creator
 *
 * This class is a factory class for clusters.
 */
class cluster_creator
{
public:
  cluster_creator( cluster_bakery const & bakery );
  ~cluster_creator();

  /**
   * @brief Create cluster object
   *
   * This method creates a cluster object that can be treated as a
   * process and added to a pipeline. It is treated the same as the
   * process constructor.
   *
   * @return New process object.
   */
  process_t operator()( kwiver::vital::config_block_sptr const& config ) const;

  cluster_bakery const m_bakery;

private:
  kwiver::vital::config_block_sptr m_default_config;

  kwiver::vital::logger_handle_t m_logger;
};

} // end namespace sprokit

#endif /* SPROKIT_PIPELINE_UTIL_CLUSTER_CREATOR_H */
