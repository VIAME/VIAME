// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core match_set interface
 */

#ifndef VITAL_MATCH_SET_H_
#define VITAL_MATCH_SET_H_

#include <vital/vital_config.h>

#include <vector>
#include <memory>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Index pair indicating matching features between two arrays
typedef std::pair< unsigned, unsigned > match;

/// A collection of matching indices between one set of objects and another.
class match_set
{
public:
  /// Destructor
  virtual ~match_set() = default;

  /// Return the number of matches in the set
  virtual size_t size() const = 0;

  /// Return a vector of matching indices
  virtual std::vector< match > matches() const = 0;
};

/// Shared pointer of base match_set type
typedef std::shared_ptr< match_set > match_set_sptr;

// ------------------------------------------------------------------
/// A concrete match set that simply wraps a vector of matches.
class simple_match_set :
  public match_set
{
public:
  /// Default Constructor
  simple_match_set() { }

  /// Constructor from a vector of matches
  explicit simple_match_set( const std::vector< match >& matches )
    : data_( matches ) { }

  /// Return the number of matches in the set
  virtual size_t size() const { return data_.size(); }

  /// Return a vector of match shared pointers
  virtual std::vector< match > matches() const { return data_; }

protected:
  /// The vector of matches
  std::vector< match > data_;
};

} } // end namespace vital

#endif // VITAL_MATCH_SET_H_
