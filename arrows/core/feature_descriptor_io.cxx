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
 * \brief Core feature_descriptor_io implementation
 */

#include "feature_descriptor_io.h"

#include <fstream>

#include <vital/logger/logger.h>
#include <vital/exceptions.h>
#include <cereal/archives/portable_binary.hpp>


using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {


/// Private implementation class
class feature_descriptor_io::priv
{
public:
  /// Constructor
  priv()
  : write_binary(false),
    m_logger( vital::get_logger( "arrows.vxl.feature_descriptor_io" ) )
  {
  }

  priv(const priv& other)
  : write_binary(other.write_binary),
    m_logger(other.m_logger)
  {
  }

  bool write_binary;

  vital::logger_handle_t m_logger;
};



/// Constructor
feature_descriptor_io
::feature_descriptor_io()
: d_(new priv)
{
}


/// Copy Constructor
feature_descriptor_io
::feature_descriptor_io(const feature_descriptor_io& other)
: d_(new priv(*other.d_))
{
}


/// Destructor
feature_descriptor_io
::~feature_descriptor_io()
{
}



/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
feature_descriptor_io
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::feature_descriptor_io::get_configuration();

  config->set_value("write_binary", d_->write_binary,
                    "Write the output data in binary format");

  return config;
}


/// Set this algorithm's properties via a config block
void
feature_descriptor_io
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated vital::config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d_->write_binary = config->get_value<bool>("write_binary",
                                             d_->write_binary);
}


/// Check that the algorithm's currently configuration is valid
bool
feature_descriptor_io
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}

namespace {

// Helper function to serialized a vector of features of known type
template <typename Archive, typename T>
void
save_features(Archive & ar, std::vector<feature_sptr> const& features)
{
  VITAL_FOREACH( const feature_sptr f, features )
  {
    if( !f )
    {
      throw vital::invalid_data("not able to write a Null feature");
    }
    if( auto ft = std::dynamic_pointer_cast<feature_<T> >(f) )
    {
      ar( *ft );
    }
    else
    {
      ar( feature_<T>(*f) );
    }
  }
}


// Helper function to unserialized a vector of N features of known type
template <typename Archive, typename T>
vital::feature_set_sptr
read_features(Archive & ar, size_t num_feat)
{
  std::vector<feature_sptr> features;
  for( size_t i=0; i<num_feat; ++i )
  {
    auto f = std::make_shared<feature_<T> >();
    ar( *f );
    features.push_back(f);
  }
  return std::make_shared<vital::simple_feature_set>(features);
}

}

/// Implementation specific load functionality.
void
feature_descriptor_io
::load_(std::string const& filename,
        vital::feature_set_sptr& feat,
        vital::descriptor_set_sptr& desc) const
{
  // open input file
  std::ifstream ifile( filename.c_str(), std::ios::binary);
  typedef cereal::PortableBinaryInputArchive Archive_t;
  Archive_t ar( ifile );

  // read "magic numbers" to validate this file as a KWIVER feature descriptor file
  int8_t file_id[4] = {0};
  ar( file_id[0], file_id[1], file_id[2], file_id[3] );
  if (std::strncmp(reinterpret_cast<char *>(file_id), "KWFD", 4) != 0)
  {
    throw vital::invalid_data("Does not look like a KWIVER feature/descriptor file: "
                              + filename);
  }
  // file format version
  uint16_t version;
  ar( version );
  if( version != 1 )
  {
    std::stringstream ss;
    ss << "Unknown file format version " << static_cast<int32_t>(version);
    throw vital::invalid_data( ss.str() );
  }

  cereal::size_type num_feat = 0;
  ar( cereal::make_size_tag(num_feat) );
  if( num_feat > 0 )
  {
    uint8_t precision;
    ar( precision );
    switch( precision )
    {
      case 32:
        feat = read_features<Archive_t, float>(ar, num_feat);
      case 64:
        feat = read_features<Archive_t, double>(ar, num_feat);
      default:
        {
          std::stringstream ss;
          ss << "unknown feature precision: " << static_cast<int32_t>(precision);
          throw vital::invalid_data("unknown feature precision: ");
        }
    }
  }
}



/// Implementation specific save functionality.
void
feature_descriptor_io
::save_(std::string const& filename,
        vital::feature_set_sptr feat,
        vital::descriptor_set_sptr desc) const
{
  if( !(feat && feat->size() > 0) &&
      !(desc && desc->size() > 0) )
  {
    LOG_WARN(d_->m_logger, "Not writing file, no features or descriptors");
    return;
  }

  // open output file
  std::ofstream ofile( filename.c_str(), std::ios::binary);
  typedef cereal::PortableBinaryOutputArchive Archive_t;
  Archive_t ar( ofile );

  // write "magic numbers" to identify this file as a KWIVER feature descriptor file
  const int8_t file_id[] = "KWFD";
  ar( file_id[0], file_id[1], file_id[2], file_id[3] );
  // file format version
  uint16_t version = 1;
  ar( version );

  if( feat && feat->size() > 0 )
  {
    std::vector<feature_sptr> features = feat->features();

    ar( cereal::make_size_tag( static_cast<cereal::size_type>(features.size()) ) ); // number of elements
    uint8_t precision;
    if( features[0]->data_type() == typeid(float) )
    {
      precision = 32;
    }
    else if( features[0]->data_type() == typeid(double) )
    {
      precision = 64;
    }
    else
    {
      throw vital::invalid_data("features must be float or double");
    }
    ar( precision );
    switch( precision )
    {
      case 32:
        save_features<Archive_t, float>(ar, features);
        break;
      case 64:
        save_features<Archive_t, double>(ar, features);
        break;
      default:
        throw vital::invalid_data("features must be float or double");
    }
  }
  else
  {
    ar( cereal::make_size_tag( static_cast<cereal::size_type>(0) ) ); // number of elements
  }

  //std::vector<descriptor_sptr> descriptors = desc->descriptors();
  //const descriptor_sptr d = descriptors[i];
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
