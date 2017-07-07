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
