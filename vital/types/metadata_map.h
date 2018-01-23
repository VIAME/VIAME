/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Header file for a map from frame IDs to metadata vectors
 */

#ifndef KWIVER_VITAL_METADATA_MAP_H_
#define KWIVER_VITAL_METADATA_MAP_H_

#include <vital/types/metadata.h>

#include <vital/vital_types.h>
#include <vital/vital_config.h>

#include <map>
#include <memory>

namespace kwiver {
namespace vital {

/// An abstract mapping between frame IDs and metadata vectors
/*
 * \note a vector of metadata objects is used because each frame could
 * have multiple metadata blocks.  For example, metadata may come from
 * multiple sources on a given frame or a metadata may be provided at
 * a higher sampling rate than the video sampling rate.
 */
class metadata_map
{
public:
  /// typedef for std::map from integer frame IDs to metadata vectors
  typedef std::map< frame_id_t, metadata_vector > map_metadata_t;

  /// Destructor
  virtual ~metadata_map() = default;

  /// Return the number of frames in the map
  virtual size_t size() const = 0;

  /// Return a map from integer frame IDs to metadata vectors
  virtual map_metadata_t metadata() const = 0;
};

/// typedef for a metadata shared pointer
typedef std::shared_ptr< metadata_map > metadata_map_sptr;


/// A concrete metadata_map that simply wraps a std::map.
class simple_metadata_map :
  public metadata_map
{
public:
  /// Default Constructor
  simple_metadata_map() { }

  /// Constructor from a std::map of metadata
  explicit simple_metadata_map( map_metadata_t const& metadata )
    : data_( metadata ) { }

  /// Return the number of metadata in the map
  virtual size_t size() const { return data_.size(); }

  /// Return a map from integer IDs to metadata shared pointers
  virtual map_metadata_t metadata() const { return data_; }


protected:
  /// The map from integer IDs to metadata shared pointers
  map_metadata_t data_;
};

}} // end namespace vital

#endif // KWIVER_VITAL_METADATA_MAP_H_
