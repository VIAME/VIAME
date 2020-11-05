/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * \brief Interface for serialize_metadata \link kwiver::vital::algo::algorithm_def algorithm
 *        definition \endlink.
 */

#ifndef VITAL_ARROWS_SERIALIZATION_JSON_SERIALIZE_METADATA_H_
#define VITAL_ARROWS_SERIALIZATION_JSON_SERIALIZE_METADATA_H_

#include <arrows/serialize/json/kwiver_serialize_json_export.h>

#include <vital/algo/serialize_metadata.h>
#include <vital/vital_config.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

/// An abstract base class for reading and writing metadata maps
/**
 * This class represents an abstract interface for reading and writing
 * video metadata.
 *
 * A note about the basic capabilities:
 *
 */
// KWIVER_ALGO_SERIALIZE_EXPORT
class KWIVER_SERIALIZE_JSON_EXPORT serialize_metadata
  : public vital::algo::serialize_metadata
{
public:
  /// Constructor
  serialize_metadata();

  /// Destructor
  ~serialize_metadata();

  /// Implementation specific load functionality.
  /**
   * Concrete implementations of serialize_metadata class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file the load
   * \returns an image container refering to the loaded image
   */
  virtual kwiver::vital::metadata_map_sptr load_(std::string const& filename) const;

  /// Implementation specific save functionality.
  /**
   * Concrete implementations of serialize_metadata class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file to save
   * \param data the image container refering to the image to write
   */
  virtual void save_(std::string const& filename,
                     kwiver::vital::metadata_map_sptr data) const;

private:
  class priv;
  std::unique_ptr<priv> d_;

};


} } } } // end namespace

#endif // VITAL_ALGO_SERIALIZE_METADATA_H_
