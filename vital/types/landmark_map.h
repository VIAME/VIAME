// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header file for a map from IDs to landmarks
 */

#ifndef VITAL_LANDMARK_MAP_H_
#define VITAL_LANDMARK_MAP_H_

#include "landmark.h"

#include <vital/vital_types.h>

#include <map>
#include <memory>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// An abstract mapping between track IDs and landmarks
class landmark_map
{
public:
  /// typedef for std::map from integer IDs to landmarks
  typedef std::map< landmark_id_t, landmark_sptr > map_landmark_t;

  /// Destructor
  virtual ~landmark_map() = default;

  /// Return the number of landmarks in the map
  virtual size_t size() const = 0;

  /// Return a map from integer IDs to landmark shared pointers
  virtual map_landmark_t landmarks() const = 0;
};

/// typedef for a landmark shared pointer
typedef std::shared_ptr< landmark_map > landmark_map_sptr;

// ------------------------------------------------------------------
/// A concrete landmark_map that simply wraps a std::map.
class simple_landmark_map :
  public landmark_map
{
public:
  /// Default Constructor
  simple_landmark_map() { }

  /// Constructor from a std::map of landmarks
  explicit simple_landmark_map( const map_landmark_t& landmarks )
    : data_( landmarks ) { }

  /// Return the number of landmarks in the map
  virtual size_t size() const { return data_.size(); }

  /// Return a map from integer IDs to landmark shared pointers
  virtual map_landmark_t landmarks() const { return data_; }

protected:
  /// The map from integer IDs to landmark shared pointers
  map_landmark_t data_;
};

} } // end namespace vital

#endif // VITAL_LANDMARK_MAP_H_
