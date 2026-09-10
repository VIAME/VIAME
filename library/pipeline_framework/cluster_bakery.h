// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   cluster_bakery.h
 * @brief  Interface to cluster_bakery class.
 */

#ifndef SPROKIT_PIPELINE_UTIL_CLUSTER_BAKERY_H
#define SPROKIT_PIPELINE_UTIL_CLUSTER_BAKERY_H

#include "bakery_base.h"

#include <optional>
#include <vector>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief Cluster bakery
 *
 * This class contains the internal representation of a cluster built
 * from a cluster definition.
 */

class cluster_bakery
  : public bakery_base
{
public:
  cluster_bakery();
  ~cluster_bakery() = default;

  using bakery_base::operator();
  void operator()( cluster_pipe_block const& cluster_block_ );

  // --------------------------
  class cluster_component_info_t
  {
  public:
    cluster_component_info_t();
    ~cluster_component_info_t();

    typedef std::vector< cluster_config_t > config_maps_t;
    typedef std::vector< cluster_input_t > input_maps_t;
    typedef std::vector< cluster_output_t > output_maps_t;

    config_maps_t m_configs;
    input_maps_t m_inputs;
    output_maps_t m_outputs;
  };
  typedef std::optional< cluster_component_info_t > opt_cluster_component_info_t;

  /// Name of the cluster.
  process::type_t m_type;

  /// Description of the cluster.
  process::description_t m_description;

  opt_cluster_component_info_t m_cluster;
};

using cluster_bakery_sptr = std::shared_ptr< cluster_bakery >;

} // end namespace sprokit

#endif /* SPROKIT_PIPELINE_UTIL_CLUSTER_BAKERY_H */
