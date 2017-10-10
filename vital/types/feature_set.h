/*ckwg +29
 * Copyright 2013-2014 by Kitware, Inc.
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
