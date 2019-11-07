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
#include <cmath>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {


class nonmax_suppressor
{
public:
  // constructor
  nonmax_suppressor(const double suppression_radius,
                    Eigen::AlignedBox<double, 2> feat_bbox,
                    const double scale_min,
                    const unsigned int scale_steps,
                    const unsigned int resolution)
    : m_resolution(resolution),
      m_radius(suppression_radius / resolution),
      m_feat_bbox(feat_bbox),
      m_offset(-(feat_bbox.min() / m_radius).array() + 0.5),
      m_range((feat_bbox.sizes() / m_radius).array() + 0.5),
      m_scale_min(scale_min)
  {
    masks.reserve(scale_steps);
    disks.reserve(scale_steps);
    for (unsigned s = 0; s < scale_steps; ++s)
    {
      const size_t pad = 2 * resolution + 1;
      const size_t w = (static_cast<size_t>(m_range[0]) >> s) + pad;
      const size_t h = (static_cast<size_t>(m_range[1]) >> s) + pad;
      masks.push_back(image_of<bool>(w, h));
      // set all pixels in the mask to false
      transform_image(masks.back(), [](bool) { return false; });

      // create the offsets for each pixel within the circle diameter
      disks.push_back(compute_disk_offsets(resolution,
                                           masks.back().w_step(),
                                           masks.back().h_step()));
    }
  }

  // --------------------------------------------------------------------------
  // test if a feature is already covered and if not, cover it
  // return true if the feature was covered by this call
  bool cover(feature const& feat)
  {
    // compute the scale and location bin indices
    unsigned scale = static_cast<unsigned>(std::log2(feat.scale()) - m_scale_min);
    vector_2i bin_idx = (feat.loc() / m_radius + m_offset).cast<int>();

    // get the center bin at this location from the mask at the current scale
    bool& bin = masks[scale]((bin_idx[0] >> scale) + m_resolution,
                             (bin_idx[1] >> scale) + m_resolution);
    // if the feature is at an uncovered location
    if (!bin)
    {
      // mark all points in a circular neighborhood as covered
      for (auto const& ptr_offset : disks[scale])
      {
        *(&bin + ptr_offset) = true;
      }
      return true;
    }
    return false;
  }

  // --------------------------------------------------------------------------
  // uncover all pixels in the suppression masks
  void uncover_all()
  {
    for (auto& mask : masks)
    {
      transform_image(mask, [](bool) { return false; });
    }
  }

  void set_radius(double r)
  {
    m_radius = r / m_resolution;
    m_offset = -(m_feat_bbox.min() / m_radius).array() + 0.5;
    m_range = (m_feat_bbox.sizes() / m_radius).array() + 0.5;

    auto scale_steps = masks.size();
    for (unsigned s = 0; s < scale_steps; ++s)
    {
      const size_t pad = 2 * m_resolution + 1;
      const size_t w = (static_cast<size_t>(m_range[0]) >> s) + pad;
      const size_t h = (static_cast<size_t>(m_range[1]) >> s) + pad;
      masks[s].set_size(w, h, 1);

      // set all pixels in the mask to false
      transform_image(masks[s], [](bool) { return false; });

      // create the offsets for each pixel within the circle diameter
      disks[s] = compute_disk_offsets(m_resolution,
                                      masks[s].w_step(),
                                      masks[s].h_step());
    }
  }

private:
  // --------------------------------------------------------------------------
  std::vector<ptrdiff_t>
  compute_disk_offsets(unsigned int radius,
                       ptrdiff_t w_step, ptrdiff_t h_step) const
  {
    std::vector<ptrdiff_t> disk;
    const int r = static_cast<int>(radius);
    const int r2 = r * r;
    for (int j = -r; j <= r; ++j)
    {
      const int j2 = j * j;
      for (int i = -r; i <= r; ++i)
      {
        const int i2 = i * i;
        if ((i2 + j2) > r2)
        {
          continue;
        }
        disk.push_back(j*h_step + i*w_step);
      }
    }
    return disk;
  }

  std::vector<image_of<bool> > masks;
  std::vector<std::vector<ptrdiff_t> > disks;
  unsigned int m_resolution;
  double m_radius;
  Eigen::AlignedBox<double, 2> m_feat_bbox;
  vector_2d m_offset;
  vector_2d m_range;
  double m_scale_min;
};

/// Private implementation class
class filter_features_nonmax::priv
{
public:
  /// Constructor
  priv()
    : suppression_radius(0),
      resolution(3),
      num_features_target(500),
      num_features_range(50)
  {
  }

  // --------------------------------------------------------------------------
  feature_set_sptr
  filter(feature_set_sptr feat, std::vector<unsigned int> &ind) const
  {
    const std::vector<feature_sptr> &feat_vec = feat->features();

    if (feat_vec.size() <= num_features_target)
    {
      return feat;
    }

    //  Create a new vector with the index and magnitude for faster sorting
    using ud_pair = std::pair<unsigned int, double>;
    std::vector<ud_pair> indices;
    indices.reserve(feat_vec.size());
    Eigen::AlignedBox<double, 2> bbox;
    Eigen::AlignedBox<double, 1> scale_box;
    for (unsigned int i = 0; i < feat_vec.size(); i++)
    {
      auto const& feat = feat_vec[i];
      indices.push_back(std::make_pair(i, feat->magnitude()));
      bbox.extend(feat->loc());
      scale_box.extend(Eigen::Matrix<double,1,1>(feat->scale()));
    }

    const double scale_min = std::log2(scale_box.min()[0]);
    const double scale_range = std::log2(scale_box.max()[0]) - scale_min;
    const unsigned scale_steps = static_cast<unsigned>(scale_range+1);
    LOG_DEBUG(m_logger, "Using " << scale_steps << " scale steps");
    if (scale_steps > 20)
    {
      LOG_ERROR(m_logger, "Scale range is too large.  Log2 scales from "
                          << scale_box.min() << " to " << scale_box.max());
      return nullptr;
    }

    if (!bbox.sizes().allFinite())
    {
      LOG_ERROR(m_logger, "Not all features are finite");
      return nullptr;
    }

    // sort on descending feature magnitude
    std::sort(indices.begin(), indices.end(),
              [](const ud_pair &l, const ud_pair &r)
              {
                return l.second > r.second;
              });

    // compute an upper bound on the radius
    const double& w = bbox.sizes()[0];
    const double& h = bbox.sizes()[1];
    const double wph = w + h;
    const double m = num_features_target - 1;
    double high_radius = (wph + std::sqrt(wph*wph + 4 * m*w*h)) / (2 * m);
    double low_radius = 0.0;

    // initial guess for radius, if not specified
    if (suppression_radius <= 0.0)
    {
      suppression_radius = high_radius / 2.0;
    }

    nonmax_suppressor suppressor(suppression_radius,
                                 bbox, scale_min, scale_steps,
                                 resolution);

    // binary search of radius to find the target number of features
    std::vector<feature_sptr> filtered;
    while (true)
    {
      ind.clear();
      filtered.clear();
      filtered.reserve(indices.size());
      ind.reserve(indices.size());
      // check each feature against the masks to see if that location
      // has already been covered
      for (auto const& p : indices)
      {
        unsigned int index = p.first;
        auto const& feat = feat_vec[index];
        if (suppressor.cover(*feat))
        {
          // add this feature to the accepted list
          ind.push_back(index);
          filtered.push_back(feat);
        }
      }
      // if not using a target number of features, keep this result
      if (num_features_target == 0)
      {
        break;
      }

      // adjust the bounds to continue binary search
      if (filtered.size() < num_features_target)
      {
        high_radius = suppression_radius;
      }
      else if (filtered.size() > num_features_target + num_features_range)
      {
        low_radius = suppression_radius;
      }
      else
      {
        // in the valid range, so we are done
        break;
      }
      double new_suppression_radius = (high_radius + low_radius) / 2;
      if (new_suppression_radius < 0.25)
      {
        LOG_DEBUG(m_logger, "Found " << filtered.size() << " features.  "
                            "Suppression radius is too small to continue.");
        break;
      }
      suppression_radius = new_suppression_radius;
      suppressor.set_radius(suppression_radius);
      LOG_DEBUG(m_logger, "Found " << filtered.size() << " features.  "
                          "Changing suppression radius to "
                          << suppression_radius);
    }

    LOG_INFO(m_logger, "Reduced " << feat_vec.size() << " features to "
                       << filtered.size() << " features with non-max radius "
                       << suppression_radius);

    return std::make_shared<vital::simple_feature_set>(
      vital::simple_feature_set(filtered));
  }


  // configuration paramters
  mutable double suppression_radius;
  unsigned int resolution;
  unsigned int num_features_target;
  unsigned int num_features_range;
  vital::logger_handle_t m_logger;

private:

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
                    "suppress weaker features.  This is an initial guess. "
                    "The radius is adapted to reach the desired number of "
                    "features.  If target_num_features is 0 then this radius "
                    "is not adapted.");

  config->set_value("num_features_target", d_->num_features_target,
                    "The target number of features to detect. "
                    "The suppression radius is dynamically adjusted to "
                    "acheive this number of features.");

  config->set_value("num_features_range", d_->num_features_range,
                    "The number of features above target_num_features to "
                    "allow in the output.  This window allows the binary "
                    "search on radius to terminate sooner.");

  config->set_value("resolution", d_->resolution,
                    "The resolution (N) of the filter for computing neighbors."
                    " The filter is an (2N+1) x (2N+1) box containing a circle"
                    " of radius N. The value must be a positive integer. "
                    "Larger values are more "
                    "accurate at the cost of more memory and compute time.");

  return config;
}


// ----------------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
filter_features_nonmax
::set_configuration(vital::config_block_sptr config)
{
#define GET_VALUE(name, type) \
  d_->name = config->get_value<type>(#name, d_->name)

  GET_VALUE(suppression_radius, double);
  GET_VALUE(resolution, unsigned int);
  GET_VALUE(num_features_target, unsigned int);
  GET_VALUE(num_features_range, unsigned int);

#undef GET_VALUE
}


// ----------------------------------------------------------------------------
// Check that the algorithm's configuration vital::config_block is valid
bool
filter_features_nonmax
::check_configuration(vital::config_block_sptr config) const
{
  unsigned int resolution =
    config->get_value<unsigned int>("resolution", d_->resolution);
  if (resolution < 1)
  {
    LOG_ERROR(logger(), "resolution must be at least 1");
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
