// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "match_features.h"

#include <vector>

#include <arrows/viscl/descriptor_set.h>
#include <arrows/viscl/feature_set.h>
#include <arrows/viscl/match_set.h>
#include <arrows/viscl/utils.h>

#include <viscl/tasks/track_descr_match.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// Private implementation class
class match_features::priv
{
public:
  /// Constructor
  priv() : search_radius(200)
  {
  }

  viscl::track_descr_match matcher;
  unsigned int search_radius;
};

/// Constructor
match_features
::match_features()
: d_(new priv)
{
}

/// Destructor
match_features
::~match_features()
{
}

/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
match_features
::get_configuration() const
{
  vital::config_block_sptr config = algorithm::get_configuration();
  config->set_value("search_box_radius", d_->search_radius,
                    "Maximum pixel radius to search for kpt match.");
  return config;
}

/// Set this algorithm's properties via a config block
void
match_features
::set_configuration(vital::config_block_sptr config)
{
  unsigned int sbr = config->get_value<unsigned int>("search_box_radius",
                                                     d_->search_radius);
  d_->matcher.set_search_box_radius(sbr);
}

/// Check that the algorithm's configuration vital::config_block is valid
bool
match_features
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}

/// Match one set of features and corresponding descriptors to another
vital::match_set_sptr
match_features
::match(vital::feature_set_sptr feat1, vital::descriptor_set_sptr desc1,
        vital::feature_set_sptr feat2, vital::descriptor_set_sptr desc2) const
{
  if( !desc1 || !desc2 )
  {
    return vital::match_set_sptr();
  }

  viscl::buffer d1 = descriptors_to_viscl(*desc1);
  viscl::buffer d2 = descriptors_to_viscl(*desc2);

  vcl::feature_set::type f1 = vcl::features_to_viscl(*feat1);
  vcl::feature_set::type f2 = vcl::features_to_viscl(*feat2);

  size_t numkpts2 = feat2->size();
  viscl::buffer matches = d_->matcher.match(f1.features_, f1.kptmap_, d1,
                                            f2.features_, numkpts2, f2.kptmap_, d2);

  return vital::match_set_sptr(new match_set(matches));
}

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver
