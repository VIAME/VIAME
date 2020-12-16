// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VXL match_features_constrained algorithm implementation
 */

#include "match_features_constrained.h"

#include <vector>

#include <vital/types/feature_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/match_set.h>

#include <rsdl/rsdl_kd_tree.h>
#include <vnl/vnl_vector_fixed.h>

#include <limits>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace vxl {

// Private implementation class
class match_features_constrained::priv
{
public:
  // Constructor
  priv() :
    scale_thresh(2.0),
    angle_thresh(-1.0),
    radius_thresh(200.0)
  {
  }

  // ----------------------------------------------------------------------------
  // Compute the minimum angle between two angles in degrees
  inline static
  double angle_dist(double a1, double a2)
  {
    double d = a1 - a2;
    if (d > 180.0)
    {
      d -= 360;
    }
    if (d < -180.0)
    {
      d += 360;
    }
    return fabs(d);
  }

  void
  match(feature_set_sptr feat1, descriptor_set_sptr desc1,
        feature_set_sptr feat2, descriptor_set_sptr desc2,
        std::vector<vital::match> &matches) const
  {
    matches.clear();

    std::vector<rsdl_point> fixedpts;
    const std::vector<feature_sptr> &feat1_vec = feat1->features();
    const std::vector<feature_sptr> &feat2_vec = feat2->features();
    const std::vector<descriptor_sptr> desc1_vec( desc1->cbegin(), desc1->cend() );
    const std::vector<descriptor_sptr> desc2_vec( desc2->cbegin(), desc2->cend() );

    for (unsigned int i = 0; i < feat2_vec.size(); i++)
    {
      rsdl_point pt(3);
      pt.set_cartesian(vnl_vector_fixed<double, 3>(feat2_vec[i]->loc().data()));
      fixedpts.push_back(pt);
    }

    rsdl_kd_tree kdtree(fixedpts);

    for (unsigned int i = 0; i < feat1_vec.size(); i++)
    {
      vital::feature_sptr f1 = feat1_vec[i];

      std::vector<rsdl_point> points;
      std::vector<int> indices;
      rsdl_point query_pt(3);

      query_pt.set_cartesian(vnl_vector_fixed<double, 3>(f1->loc().data()));
      kdtree.points_in_radius(query_pt, this->radius_thresh, points, indices);

      int closest = -1;
      double closest_dist = std::numeric_limits<double>::max();
      vnl_vector<double> d1(desc1_vec[i]->as_double().data(),
                            static_cast<unsigned>(desc1_vec[i]->size()));

      for (unsigned int j = 0; j < indices.size(); j++)
      {
        int index = indices[j];
        vital::feature_sptr f2 = feat2_vec[index];
        if ((scale_thresh <= 0.0  || std::max(f1->scale(),f2->scale())/std::min(f1->scale(),f2->scale()) <= scale_thresh) &&
            (angle_thresh <= 0.0  || angle_dist(f2->angle(), f1->angle()) <= angle_thresh))
        {
          vnl_vector<double> d2(desc2_vec[index]->as_double().data(),
                                static_cast<unsigned>(desc2_vec[index]->size()));
          double dist = (d1 - d2).squared_magnitude();
          if (dist < closest_dist)
          {
            closest = index;
            closest_dist = dist;
          }
        }
      }

      if (closest >= 0)
      {
        matches.push_back(vital::match(i, closest));
      }
    }

    LOG_INFO( m_logger, "Found " << matches.size() << " matches.");
  }

  double scale_thresh;
  double angle_thresh;
  double radius_thresh;

  vital::logger_handle_t m_logger;
};

// ----------------------------------------------------------------------------
// Constructor
match_features_constrained
::match_features_constrained()
: d_(new priv)
{
  attach_logger( "arrows.vxl.match_features_constrained" );
  d_->m_logger = logger();
}

// Destructor
match_features_constrained
::~match_features_constrained()
{
}

// ----------------------------------------------------------------------------
// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
match_features_constrained
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config =
      vital::algo::match_features::get_configuration();

  config->set_value("scale_thresh", d_->scale_thresh,
                    "Ratio threshold of scales between matching keypoints (>=1.0)"
                    " -1 turns scale thresholding off");

  config->set_value("angle_thresh", d_->angle_thresh,
                    "Angle difference threshold between matching keypoints"
                    " -1 turns angle thresholding off");

  config->set_value("radius_thresh", d_->radius_thresh,
                    "Search radius for a match in pixels");

  return config;
}

// ----------------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
match_features_constrained
::set_configuration(vital::config_block_sptr config)
{
  d_->scale_thresh = config->get_value<double>("scale_thresh", d_->scale_thresh);
  d_->angle_thresh = config->get_value<double>("angle_thresh", d_->angle_thresh);
  d_->radius_thresh = config->get_value<double>("radius_thresh", d_->radius_thresh);
}

// ----------------------------------------------------------------------------
// Check that the algorithm's configuration vital::config_block is valid
bool
match_features_constrained
::check_configuration(vital::config_block_sptr config) const
{
  double radius_thresh = config->get_value<double>("radius_thresh", d_->radius_thresh);
  if (radius_thresh <= 0.0)
  {
    LOG_ERROR( logger(), "radius_thresh should be > 0.0, is " << radius_thresh);
    return false;
  }
  double scale_thresh = config->get_value<double>("scale_thresh", d_->scale_thresh);
  if (scale_thresh < 1.0 && scale_thresh >= 0.0)
  {
    LOG_ERROR( logger(), "scale_thresh should be >= 1.0 (or < 0.0 to disable), is "
                               << scale_thresh);
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
// Match one set of features and corresponding descriptors to another
vital::match_set_sptr
match_features_constrained
::match(vital::feature_set_sptr feat1, vital::descriptor_set_sptr desc1,
        vital::feature_set_sptr feat2, vital::descriptor_set_sptr desc2) const
{
  if( !feat1 || !feat2 || !desc1 || !desc2 )
  {
    return match_set_sptr();
  }

  std::vector<vital::match> matches;
  d_->match(feat1, desc1, feat2, desc2, matches);

  return std::make_shared<vital::simple_match_set>(vital::simple_match_set(matches));
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
