/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief data_serializer algorithm definition
 */

#ifndef VITAL_ALGO_SERIALIZE_H
#define VITAL_ALGO_SERIALIZE_H

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/any.h>

#include <memory>
#include <string>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class serializing and deserializing.
/**
 * This class represents a pair of methods that serialize and
 * deserialize a concrete data type. These methods are guaranteed to
 * work together. A data type serialized and deserialized by this
 * algorithm are semantically equivalent. The format and process of
 * actually doing the serialization depends on the concrete
 * implementation.
 *
 * The serializer is stateless and idempotent.
 *
 * The main application for this algorithm is to serialize data
 * objects for inter-process communications.
 */
class VITAL_ALGO_EXPORT data_serializer
  : public kwiver::vital::algorithm_def< data_serializer >
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "data_serializer"; }

  /// Serialize the item into a byte string.
  /**
   * This method serializes the supplied data item and returns a byte
   * string. The type of the data represented by the datum must match
   * the type expected by the serializer implementation or an
   * exception will be thrown.
   *
   * @param item Data item to be serialized.
   *
   * @return Byte string of serialized data item.
   *
   * @throws kwiver::vital::bad_any_cast
   */
  virtual std::shared_ptr< std::string > serialize( const kwiver::vital::any& item ) = 0;

  /// Deserialize byte string into data type.
  /**
   * Deserialize the supplied string of bytes into a new data
   * item. This method must handle an input byte string created by the
   * serialize() method and convert it to the concrete type. The
   * actual type used for the conversion is based on the concrete
   * implementation of this algorithm. If the input byte string does
   * not represent the expected data type, then an exception will be
   * thrown.
   *
   * @param message Serialized data item that is to be processed.
   *
   * @return Concrete data type, represented as an any, created from
   * the input.
   *
   * @throws kwiver::vital::bad_any_cast
   */
  virtual const kwiver::vital::any deserialize( std::shared_ptr< std::string > message ) = 0;

  virtual void set_configuration( kwiver::vital::config_block_sptr config ) { }
  virtual bool check_configuration(config_block_sptr config) const { return true; }

protected:
  data_serializer();

};

/// Shared pointer for detect_features algorithm definition class
typedef std::shared_ptr<data_serializer> data_serializer_sptr;

} // end namespaces
}
}

#endif // VITAL_ALGO_SERIALIZE_H
