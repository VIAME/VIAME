// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of filter_features_magnitude algorithm
 */
#include "filter_features_magnitude.h"

#include <algorithm>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {

//Helper struct for the filter function
struct feature_at_index_is_greater
{
  bool operator()(const std::pair<unsigned int, double> &l, const std::pair<unsigned int, double> &r)
  {
    return l.second > r.second;
  }
};

/// Private implementation class
class filter_features_magnitude::priv
{
public:
  /// Constructor
  priv()
    : top_fraction(0.2),
      min_features(100)
  {
  }

  feature_set_sptr
  filter(feature_set_sptr feat, std::vector<unsigned int> &ind) const
  {
    const std::vector<feature_sptr> &feat_vec = feat->features();
    ind.clear();
    if (feat_vec.size() <= min_features)
    {
      ind.resize(feat_vec.size());
      for (unsigned int i=0; i<ind.size(); ++i)
      {
        ind[i] = i;
      }
      return feat;
    }

    //  Create a new vector with the index and magnitude for faster sorting
    std::vector<std::pair<unsigned int, double> > indices;
    indices.reserve(feat_vec.size());
    for (unsigned int i = 0; i < feat_vec.size(); i++)
    {
      indices.push_back(std::make_pair(i, feat_vec[i]->magnitude()));
    }

    // compute threshold
    unsigned int cutoff = std::max(min_features, static_cast<unsigned int>(top_fraction * indices.size()));

    // partially sort on descending feature magnitude
    std::nth_element(indices.begin(), indices.begin()+cutoff, indices.end(),
                     feature_at_index_is_greater());

    std::vector<feature_sptr> filtered(cutoff);
    ind.resize(cutoff);
    for (unsigned int i = 0; i < cutoff; i++)
    {
      unsigned int index = indices[i].first;
      ind[i] = index;
      filtered[i] = feat_vec[index];
    }

    LOG_INFO( m_logger,
             "Reduced " << feat_vec.size() << " features to " << filtered.size() << " features.");

    return std::make_shared<vital::simple_feature_set>(vital::simple_feature_set(filtered));
  }

  double top_fraction;
  unsigned int min_features;
  vital::logger_handle_t m_logger;
};

// ----------------------------------------------------------------------------
// Constructor
filter_features_magnitude
::filter_features_magnitude()
: d_(new priv)
{
  attach_logger( "arrows.core.filter_features_magnitude" );
  d_->m_logger = logger();
}

// Destructor
filter_features_magnitude
::~filter_features_magnitude()
{
}

// ----------------------------------------------------------------------------
// Get this algorithm's \link vital::config_block configuration block \endlink
  vital::config_block_sptr
filter_features_magnitude
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config =
      vital::algo::filter_features::get_configuration();

  config->set_value("top_fraction", d_->top_fraction,
                    "Fraction of strongest keypoints to keep, range (0.0, 1.0]");

  config->set_value("min_features", d_->min_features,
                    "minimum number of features to keep");

  return config;
}

// ----------------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
filter_features_magnitude
::set_configuration(vital::config_block_sptr config)
{
  d_->top_fraction = config->get_value<double>("top_fraction", d_->top_fraction);
  d_->min_features = config->get_value<unsigned int>("min_features", d_->min_features);
}

// ----------------------------------------------------------------------------
// Check that the algorithm's configuration vital::config_block is valid
bool
filter_features_magnitude
::check_configuration(vital::config_block_sptr config) const
{
  double top_fraction = config->get_value<double>("top_fraction", d_->top_fraction);
  if( top_fraction <= 0.0 || top_fraction > 1.0 )
  {
    LOG_ERROR( logger(),
             "top_fraction parameter is " << top_fraction << ", needs to be in (0.0, 1.0].");
    return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
// Filter feature set
vital::feature_set_sptr
filter_features_magnitude
::filter(vital::feature_set_sptr feat, std::vector<unsigned int> &indices) const
{
  return d_->filter(feat, indices);
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
