/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
 * \brief Implementation of filter_features_nonmax algorithm
 */
#include "filter_features_nonmax.h"

#include <vital/types/image.h>
#include <vital/util/transform_image.h>

#include <Eigen/Geometry>
#include <algorithm>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {


/// Private implementation class
class filter_features_nonmax::priv
{
public:
  /// Constructor
  priv()
    : suppression_radius(5.0)
  {
  }

  feature_set_sptr
  filter(feature_set_sptr feat, std::vector<unsigned int> &ind) const
  {
    const std::vector<feature_sptr> &feat_vec = feat->features();
    ind.clear();

    if (feat_vec.empty())
    {
      return feat;
    }

    //  Create a new vector with the index and magnitude for faster sorting
    using ud_pair = std::pair<unsigned int, double>;
    std::vector<ud_pair> indices;
    indices.reserve(feat_vec.size());
    Eigen::AlignedBox<double, 2> bbox;
    for (unsigned int i = 0; i < feat_vec.size(); i++)
    {
      indices.push_back(std::make_pair(i, feat_vec[i]->magnitude()));
      bbox.extend(feat_vec[i]->loc());
    }

    vector_2d offset = -(bbox.min() / suppression_radius).array() + 0.5;
    vector_2d range = (bbox.sizes() / suppression_radius).array() + 0.5;
    if (range[0] > std::numeric_limits<int>::max() ||
        range[1] > std::numeric_limits<int>::max())
    {
      LOG_ERROR(m_logger, "Range of feature locations is tool large for "
                          "non-max suppression");
      return nullptr;
    }
    image_of<bool> mask(static_cast<size_t>(range[0])+1,
                        static_cast<size_t>(range[1])+1);
    // set all pixels in the mask to false
    transform_image(mask, [](bool) { return false; });

    // sort on descending feature magnitude
    std::sort(indices.begin(), indices.end(),
              [](const ud_pair &l, const ud_pair &r)
              {
                return l.second > r.second;
              });

    std::vector<feature_sptr> filtered;
    ind.reserve(indices.size());
    for (auto const& p : indices)
    {
      unsigned int index = p.first;
      auto const& feat = feat_vec[index];
      vector_2i bin_idx = (feat->loc() / suppression_radius + offset).cast<int>();
      bool& bin = mask(bin_idx[0], bin_idx[1]);
      if (!bin)
      {
        bin = true;
        ind.push_back(index);
        filtered.push_back(feat);
      }
    }

    LOG_INFO( m_logger,
             "Reduced " << feat_vec.size() << " features to " << filtered.size() << " features.");

    return std::make_shared<vital::simple_feature_set>(vital::simple_feature_set(filtered));
  }

  double suppression_radius;
  vital::logger_handle_t m_logger;
};


// ----------------------------------------------------------------------------
// Constructor
filter_features_nonmax
::filter_features_nonmax()
: d_(new priv)
{
  attach_logger( "arrows.core.filter_features_nonmax" );
  d_->m_logger = logger();
}


// Destructor
filter_features_nonmax
::~filter_features_nonmax()
{
}


// ----------------------------------------------------------------------------
// Get this algorithm's \link vital::config_block configuration block \endlink
  vital::config_block_sptr
filter_features_nonmax
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config =
      vital::algo::filter_features::get_configuration();

  config->set_value("suppression_radius", d_->suppression_radius,
                    "The radius, in pixels, within which to "
                    "suppress weaker features");

  return config;
}


// ----------------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
filter_features_nonmax
::set_configuration(vital::config_block_sptr config)
{
  d_->suppression_radius =
    config->get_value<double>("suppression_radius", d_->suppression_radius);
}


// ----------------------------------------------------------------------------
// Check that the algorithm's configuration vital::config_block is valid
bool
filter_features_nonmax
::check_configuration(vital::config_block_sptr config) const
{
  double suppression_radius =
    config->get_value<double>("suppression_radius", d_->suppression_radius);
  if( suppression_radius < 1.0)
  {
    LOG_ERROR( logger(), "suppression_radius must be at least 1.0");
    return false;
  }
  return true;
}


// ----------------------------------------------------------------------------
// Filter feature set
vital::feature_set_sptr
filter_features_nonmax
::filter(vital::feature_set_sptr feat, std::vector<unsigned int> &indices) const
{
  return d_->filter(feat, indices);
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
