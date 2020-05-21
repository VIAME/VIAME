/*ckwg +29
 * Copyright 2015-2016, 2019-2020 by Kitware, Inc.
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
 * \brief VXL match_features_constrained algorithm impl interface
 */

#ifndef KWIVER_ARROWS_VXL_MATCH_FEATURES_CONSTRAINED_H_
#define KWIVER_ARROWS_VXL_MATCH_FEATURES_CONSTRAINED_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/match_features.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// A match_feature algorithm that uses feature position, orientation, and scale constraints
/**
 *  This matching algorithm assumes that the features to be matched are already
 *  somewhat well aligned geometrically.  The use cases are very similar images
 *  (e.g. adjacent frames of video) and features that have been transformed
 *  into approximate alignment by a pre-processing step
 *  (e.g. image registration)
 *
 *  This algorithm first reduces the search space for each feature using a
 *  search radius in the space of location (and optionally orientation and
 *  scale) to find only geometrically nearby features.  It then looks at
 *  the descriptors for the neighbors and finds the best match by appearance.
 */
class KWIVER_ALGO_VXL_EXPORT match_features_constrained
  : public vital::algo::match_features
{
public:
  PLUGIN_INFO( "vxl_constrained",
               "Use VXL to match descriptors under the constraints of similar geometry "
               "(rotation, scale, position)." )

  /// Constructor
  match_features_constrained();

  /// Destructor
  virtual ~match_features_constrained();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration vital::config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Match one set of features and corresponding descriptors to another
  /**
   * \param [in] feat1 the first set of features to match
   * \param [in] desc1 the descriptors corresponding to \a feat1
   * \param [in] feat2 the second set fof features to match
   * \param [in] desc2 the descriptors corresponding to \a feat2
   * \returns a set of matching indices from \a feat1 to \a feat2
   */
  virtual vital::match_set_sptr
  match(vital::feature_set_sptr feat1, vital::descriptor_set_sptr desc1,
        vital::feature_set_sptr feat2, vital::descriptor_set_sptr desc2) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
