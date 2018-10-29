/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file   cluster_bakery.h
 * @brief  Interface to cluster_bakery class.
 */

#ifndef SPROKIT_PIPELINE_UTIL_CLUSTER_BAKERY_H
#define SPROKIT_PIPELINE_UTIL_CLUSTER_BAKERY_H

#include "bakery_base.h"

#include <vital/optional.h>

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
  ~cluster_bakery();

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
  typedef kwiver::vital::optional< cluster_component_info_t > opt_cluster_component_info_t;

  process::type_t m_type;
  process::description_t m_description;
  opt_cluster_component_info_t m_cluster;
};

} // end namespace sprokit

#endif /* SPROKIT_PIPELINE_UTIL_CLUSTER_BAKERY_H */
