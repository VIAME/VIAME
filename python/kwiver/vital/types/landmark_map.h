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

#ifndef KWIVER_VITAL_PYTHON_LANDMARK_MAP_H_
#define KWIVER_VITAL_PYTHON_LANDMARK_MAP_H_

#include <pybind11/pybind11.h>
#include <vital/vital_types.h>
#include <vital/types/landmark.h>
#include <python/kwiver/vital/types/landmark.h>

namespace py = pybind11;


namespace kwiver {
namespace vital {


/// typedef for std::map from integer IDs to landmarks
typedef std::shared_ptr<PyLandmarkBase> py_landmark_sptr;
typedef std::map< landmark_id_t, py_landmark_sptr > py_map_landmark_t;

// ------------------------------------------------------------------
/// An abstract mapping between track IDs and landmarks
class PyLandmarkMapBase
{
public:

  /// Destructor
  virtual ~PyLandmarkMapBase() = default;

  /// Return the number of landmarks in the map
  virtual size_t size() const = 0;

  /// Return a map from integer IDs to landmark shared pointers
  virtual py_map_landmark_t landmarks() const = 0;
};

/// typedef for a landmark shared pointer
typedef std::shared_ptr< PyLandmarkMapBase > py_landmark_map_sptr;


// ------------------------------------------------------------------
/// A concrete landmark_map that simply wraps a std::map.
class PyLandmarkMap :
  public PyLandmarkMapBase
{
public:
  /// Default Constructor
  PyLandmarkMap() { }

  /// Constructor from a std::map of landmarks
  explicit PyLandmarkMap( const py_map_landmark_t& landmarks )
    : data_( landmarks ) { }

  /// Return the number of landmarks in the map
  virtual size_t size() const { return data_.size(); }

  /// Return a map from integer IDs to landmark shared pointers
  virtual py_map_landmark_t landmarks() const { return data_; }


protected:
  /// The map from integer IDs to landmark shared pointers
  py_map_landmark_t data_;
};

} } // end namespace vital

typedef kwiver::vital::PyLandmarkMapBase landmark_map_t;
typedef kwiver::vital::PyLandmarkMap s_landmark_map_t;
typedef kwiver::vital::py_map_landmark_t map_landmark_t;

void landmark_map(py::module &m);

#endif // VITAL_LANDMARK_MAP_H_
