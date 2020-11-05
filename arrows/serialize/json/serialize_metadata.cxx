/*ckwg +29
 * Copyright 2013-2018 by Kitware, Inc.
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
 * \brief Implementation of load/save functionality.
 */
#include <vital/types/metadata_map.h>

#include <arrows/serialize/json/metadata.h>
#include <arrows/serialize/json/serialize_metadata.h>

#include <iostream>
#include <fstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

/// Private implementation class
class serialize_metadata::priv
{
public:
  /// Constructor
  priv() {}

  kwiver::arrows::serialize::json::metadata serializer;
};

/// Constructor
serialize_metadata::serialize_metadata() {}

/// Destructor
serialize_metadata::~serialize_metadata() {}

/// Implementation specific load functionality.
/**
 * Concrete implementations of serialize_metadata class must provide an
 * implementation for this method.
 *
 * \param filename the path to the file the load
 * \returns an image container refering to the loaded image
 */
kwiver::vital::metadata_map_sptr
serialize_metadata::load_(std::string const& filename) const
{
  return kwiver::vital::metadata_map_sptr();
}

/// Implementation specific save functionality.
/**
 * Concrete implementations of serialize_metadata class must provide an
 * implementation for this method.
 *
 * \param filename the path to the file to save
 * \param data the image container refering to the image to write
 */
// TODO see if this should throw something
void
serialize_metadata::save_(std::string const& filename,
                          kwiver::vital::metadata_map_sptr data) const
{
  auto metadata = data->metadata();
  std::shared_ptr< std::string > serialized =
    d_->serializer.serialize_map(metadata);
  std::ofstream fout( filename.c_str() );

  if( ! fout )
  {
    std::cout << "Couldn't open \"" << filename << "\" for writing.\n";
  }

  fout << *serialized << std::endl;
}

} } } } // end namespace
