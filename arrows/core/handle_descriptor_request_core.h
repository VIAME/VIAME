/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \file
 * \brief Header defining the handle_descriptor_request_core algorithm
 */

#ifndef ARROWS_PLUGINS_CORE_FORMULATE_QUERY_CORE_H_
#define ARROWS_PLUGINS_CORE_FORMULATE_QUERY_CORE_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/algorithm.h>
#include <vital/algo/handle_descriptor_request.h>

#include <vital/algo/image_io.h>
#include <vital/algo/compute_track_descriptors.h>

namespace kwiver {
namespace arrows {
namespace core {

/// A basic query formulator
class KWIVER_ALGO_CORE_EXPORT handle_descriptor_request_core
  : public vital::algorithm_impl< handle_descriptor_request_core,
      vital::algo::handle_descriptor_request >
{
public:
  /// Name of the algorithm
  static constexpr char const* name = "core";

  /// Description of the algorithm
  static constexpr char const* description =
    "Formulate descriptors for later queries.";

  /// Default Constructor
  handle_descriptor_request_core();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  /**
   * This base virtual function implementation returns an empty configuration
   * block whose name is set to \c this->type_name.
   *
   * \returns \c config_block containing the configuration for this algorithm
   *          and any nested components.
   */
  virtual vital::config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  /**
   * \throws no_such_configuration_value_exception
   *    Thrown if an expected configuration value is not present.
   * \throws algorithm_configuration_exception
   *    Thrown when the algorithm is given an invalid \c config_block or is'
   *    otherwise unable to configure itself.
   *
   * \param config  The \c config_block instance containing the configuration
   *                parameters for this algorithm
   */
  virtual void set_configuration( vital::config_block_sptr config );

  /// Check that the algorithm's currently configuration is valid
  /**
   * This checks solely within the provided \c config_block and not against
   * the current state of the instance. This isn't static for inheritence
   * reasons.
   *
   * \param config  The config block to check configuration of.
   *
   * \returns true if the configuration check passed and false if it didn't.
   */
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Formulate query
  virtual bool handle(
    kwiver::vital::descriptor_request_sptr request,
    kwiver::vital::track_descriptor_set_sptr& desc,
    std::vector< kwiver::vital::image_container_sptr >& imgs );


private:

  /// The feature detector algorithm to use
  vital::algo::image_io_sptr reader_;

  /// The descriptor extractor algorithm to use
  vital::algo::compute_track_descriptors_sptr extractor_;
};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif // KWIVER_ARROWS_CORE_FORMULATE_QUERY_CORE_H_
