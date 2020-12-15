// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core feature_set class interface
 */

#ifndef VITAL_FEATURE_SET_H_
#define VITAL_FEATURE_SET_H_

#include "feature.h"

#include <vital/vital_config.h>

#include <vector>

namespace kwiver {
namespace vital {

/// An abstract ordered collection of 2D image feature points.
/**
 * The base class feature_set is abstract and provides an interface for
 * returning a vector of features.  There is a simple derived class that
 * stores the data as a vector of features and returns it.  Other derived
 * classes can store the data in other formats and convert on demand.
 */
class feature_set
{
public:
  /// Destructor
  virtual ~feature_set() = default;

  /// Return the number of features in the set
  virtual size_t size() const = 0;

  /// Return a vector of feature shared pointers
  virtual std::vector< feature_sptr > features() const = 0;
};

/// Shared pointer for base feature_set type
typedef std::shared_ptr< feature_set > feature_set_sptr;

/// A concrete feature set that simply wraps a vector of features.
class simple_feature_set :
  public feature_set
{
public:
  /// Default Constructor
  simple_feature_set() { }

  /// Constructor from a vector of features
  explicit simple_feature_set( const std::vector< feature_sptr >& features )
    : data_( features ) { }

  /// Return the number of feature in the set
  virtual size_t size() const { return data_.size(); }

  /// Return a vector of feature shared pointers
  virtual std::vector< feature_sptr > features() const { return data_; }

protected:
  /// The vector of features
  std::vector< feature_sptr > data_;
};

} } // end namespace vital

#endif // VITAL_FEATURE_SET_H_
