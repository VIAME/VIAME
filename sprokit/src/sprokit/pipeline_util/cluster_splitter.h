// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   cluster_splitter.h
 * @brief  Interface for cluster_splitter class
 */

#ifndef SPROKIT_PIPELINE_UTIL_CLUSTER_SPLITTER_H
#define SPROKIT_PIPELINE_UTIL_CLUSTER_SPLITTER_H

#include "cluster_bakery.h"

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief Separate cluster blocks by type.
 *
 * This class/visitor separates each cluster block type into its own
 * list.
 */
  class cluster_splitter
  {
  public:
    cluster_splitter(cluster_bakery::cluster_component_info_t& info);
    ~cluster_splitter();

    void operator () (cluster_config_t const& config_block);
    void operator () (cluster_input_t const& input_block);
    void operator () (cluster_output_t const& output_block);

    // is filled in by visitor.
    cluster_bakery::cluster_component_info_t& m_info;

  private:
    typedef std::set<process::port_t> unique_ports_t;

    unique_ports_t m_input_ports;
    unique_ports_t m_output_ports;
  };

} // end namespace sprokit

#endif /* SPROKIT_PIPELINE_UTIL_CLUSTER_SPLITTER_H */
