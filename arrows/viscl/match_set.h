// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VISCL_MATCH_SET_H_
#define KWIVER_ARROWS_VISCL_MATCH_SET_H_

#include <vital/vital_config.h>
#include <arrows/viscl/kwiver_algo_viscl_export.h>

#include <vital/types/match_set.h>

#include <viscl/core/buffer.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// A concrete match set that wraps VisCL matches
class KWIVER_ALGO_VISCL_EXPORT match_set
: public vital::match_set
{
public:
  /// Default constructor
  match_set() {}

  /// Constructor from VisCL matches
  explicit match_set(const viscl::buffer& viscl_matches)
   : data_(viscl_matches) {}

  /// Return the number of matches in the set
  /**
    * Warning: this function is slow, it downloads all of the matches
    * to count them it is recommended to use matches() if you need both
    * the size and the matches.
    */
  virtual size_t size() const;

  /// Return a vector of matching indices
  virtual std::vector<vital::match> matches() const;

  /// Return the underlying VisCL match data
  const viscl::buffer& viscl_matches() const { return data_; }

private:
  // The collection of VisCL match data
  viscl::buffer data_;
};

/// Convert any match set to VisCL match data
/**
  * Will remove duplicate matches to a kpt from 2nd set
  */
KWIVER_ALGO_VISCL_EXPORT viscl::buffer
matches_to_viscl(const vital::match_set& match_set);

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver

#endif
