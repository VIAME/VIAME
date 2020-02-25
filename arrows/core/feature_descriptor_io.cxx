/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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

#include <vital/exceptions.h>
#include <cereal/archives/portable_binary.hpp>


using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {


// Private implementation class
class feature_descriptor_io::priv
{
public:
  /// Constructor
  priv()
  : write_float_features(false)
  {
  }

  bool write_float_features;
};



// Constructor
feature_descriptor_io
::feature_descriptor_io()
: d_(new priv)
{
  attach_logger( "arrows.core.feature_descriptor_io" );
}


// Destructor
feature_descriptor_io
::~feature_descriptor_io()
{
}


// ----------------------------------------------------------------------------
// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
feature_descriptor_io
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::feature_descriptor_io::get_configuration();

  config->set_value("write_float_features", d_->write_float_features,
                    "Convert features to use single precision floats "
                    "instead of doubles when writing to save space");

  return config;
}


// ----------------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
feature_descriptor_io
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated vital::config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d_->write_float_features = config->get_value<bool>("write_float_features",
                                                     d_->write_float_features);
}


// ----------------------------------------------------------------------------
// Check that the algorithm's currently configuration is valid
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
  for( const feature_sptr f : features )
  {
    if( !f )
    {
      VITAL_THROW( vital::invalid_data,"not able to write a Null feature");
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


// ----------------------------------------------------------------------------
// Helper function to unserialized a vector of N features of known type
template <typename Archive, typename T>
vital::feature_set_sptr
read_features(Archive & ar, size_t num_feat)
{
  std::vector<feature_sptr> features;
  features.reserve(num_feat);
  for( size_t i=0; i<num_feat; ++i )
  {
    auto f = std::make_shared<feature_<T> >();
    ar( *f );
    features.push_back(f);
  }
  return std::make_shared<vital::simple_feature_set>(features);
}


// ----------------------------------------------------------------------------
// Helper function to serialized a vector of descriptors of known type
template <typename Archive, typename T>
void
save_descriptors(Archive & ar, descriptor_set_sptr const& descriptors)
{
  // dimensionality of each descriptor
  cereal::size_type dim = descriptors->at(0)->size();
  ar( cereal::make_size_tag( dim ) );
  for( descriptor_sptr const d : *descriptors )
  {
    if( !d )
    {
      VITAL_THROW( vital::invalid_data, "not able to write a Null descriptor");
    }
    if( d->size() != dim )
    {
      VITAL_THROW( vital::invalid_data, std::string("descriptor dimension is not ")
                                + "consistent, should be " + std::to_string(dim)
                                + ", is " + std::to_string(d->size()));
    }
    if( auto dt = std::dynamic_pointer_cast<descriptor_array_of<T> >(d) )
    {
      const T* data = dt->raw_data();
      for(unsigned i=0; i<dim; ++i, ++data)
      {
        ar( *data );
      }
    }
    else
    {
      VITAL_THROW( vital::invalid_data, std::string("saving descriptors of type ")
                                + typeid(T).name() + " but received type "
                                + d->data_type().name());
    }
  }
}


// ----------------------------------------------------------------------------
// Helper function to unserialized a vector of N descriptors of known type
template <typename Archive, typename T>
vital::descriptor_set_sptr
read_descriptors(Archive & ar, size_t num_desc)
{
  // dimensionality of each descriptor
  cereal::size_type dim;
  ar( cereal::make_size_tag( dim ) );

  std::vector<descriptor_sptr> descriptors;
  descriptors.reserve(num_desc);
  for( size_t i=0; i<num_desc; ++i )
  {
    std::shared_ptr<descriptor_array_of<T> > d;
    // allocate fixed vectors for common dimensions
    switch( dim )
    {
      case 128:
        d = std::make_shared<descriptor_fixed<T,128> >();
        break;
      case 64:
        d = std::make_shared<descriptor_fixed<T,64> >();
        break;
      default:
        d = std::make_shared<descriptor_dynamic<T> >(dim);
    }
    T* data = d->raw_data();
    for(unsigned i=0; i<dim; ++i, ++data)
    {
      ar( *data );
    }
    descriptors.push_back(d);
  }
  return std::make_shared<vital::simple_descriptor_set>(descriptors);
}


// ----------------------------------------------------------------------------
// compute base 2 log of integers at compile time
constexpr size_t log2(size_t n)
{
  return ( (n<2) ? 0 : 1+log2(n/2));
}


// ----------------------------------------------------------------------------
// compute a unique byte code for built-in types
template <typename T>
struct type_traits
{
  constexpr static uint8_t code = static_cast<uint8_t>(
    (std::numeric_limits<T>::is_integer << 5) +
    (std::numeric_limits<T>::is_signed << 4) +
    log2(sizeof(T)));
};


// ----------------------------------------------------------------------------
uint8_t code_from_typeid(std::type_info const& tid)
{
#define CODE_TYPE(T) \
  if(tid == typeid(T))           \
  {                              \
    return type_traits<T>::code; \
  }

  CODE_TYPE(uint8_t);
  CODE_TYPE(int8_t);
  CODE_TYPE(uint16_t);
  CODE_TYPE(int16_t);
  CODE_TYPE(uint32_t);
  CODE_TYPE(int32_t);
  CODE_TYPE(uint64_t);
  CODE_TYPE(int64_t);
  CODE_TYPE(float);
  CODE_TYPE(double);

#undef CODE_TYPE
  return 0;
}

}


// ----------------------------------------------------------------------------
// Implementation specific load functionality.
void
feature_descriptor_io
::load_(std::string const& filename,
        vital::feature_set_sptr& feat,
        vital::descriptor_set_sptr& desc) const
{
  // open input file
  std::ifstream ifile( filename.c_str(), std::ios::binary);

  // read "magic numbers" to validate this file as a KWIVER feature descriptor file
  char file_id[5] = {0};
  ifile.read(file_id, 4);
  if (std::strncmp(file_id, "KWFD", 4) != 0)
  {
    VITAL_THROW( vital::invalid_data, "Does not look like a KWIVER feature/descriptor file: "
                              + filename);
  }

  typedef cereal::PortableBinaryInputArchive Archive_t;
  Archive_t ar( ifile );

  // file format version
  uint16_t version;
  ar( version );
  if( version != 1 )
  {
    VITAL_THROW( vital::invalid_data, "Unknown file format version: "
                               + std::to_string(version) );
  }

  cereal::size_type num_feat = 0;
  ar( cereal::make_size_tag(num_feat) );
  if( num_feat > 0 )
  {
    uint8_t type_code;
    ar( type_code );
    switch( type_code )
    {
      case type_traits<float>::code:
        feat = read_features<Archive_t, float>(ar, num_feat);
        break;
      case type_traits<double>::code:
        feat = read_features<Archive_t, double>(ar, num_feat);
        break;
      default:
        VITAL_THROW( vital::invalid_data, "unknown feature type code: "
                                  + std::to_string(type_code));
    }
  }
  else
  {
    feat = feature_set_sptr();
  }

  cereal::size_type num_desc = 0;
  ar( cereal::make_size_tag(num_desc) );
  if( num_desc > 0 )
  {
    uint8_t type_code;
    ar( type_code );
    switch( type_code )
    {
#define DO_CASE(T)                                           \
      case type_traits<T>::code:                             \
        desc = read_descriptors<Archive_t, T>(ar, num_desc); \
        break

      DO_CASE(uint8_t);
      DO_CASE(int8_t);
      DO_CASE(uint16_t);
      DO_CASE(int16_t);
      DO_CASE(uint32_t);
      DO_CASE(int32_t);
      DO_CASE(uint64_t);
      DO_CASE(int64_t);
      DO_CASE(float);
      DO_CASE(double);
#undef DO_CASE

      default:
        VITAL_THROW( vital::invalid_data, "unknown descriptor type code: "
                                  + std::to_string(type_code));
    }
  }
  else
  {
    desc = descriptor_set_sptr();
  }
}


// ----------------------------------------------------------------------------
// Implementation specific save functionality.
void
feature_descriptor_io
::save_(std::string const& filename,
        vital::feature_set_sptr feat,
        vital::descriptor_set_sptr desc) const
{
  if( !(feat && feat->size() > 0) &&
      !(desc && desc->size() > 0) )
  {
    LOG_WARN(logger(), "Not writing file, no features or descriptors");
    return;
  }

  // open output file
  std::ofstream ofile( filename.c_str(), std::ios::binary);
  // write "magic numbers" to identify this file as a KWIVER feature descriptor file
  ofile.write("KWFD", 4);

  typedef cereal::PortableBinaryOutputArchive Archive_t;
  Archive_t ar( ofile );

  // file format version
  uint16_t version = 1;
  ar( version );

  if( feat && feat->size() > 0 )
  {
    std::vector<feature_sptr> features = feat->features();

    ar( cereal::make_size_tag( static_cast<cereal::size_type>(features.size()) ) ); // number of elements
    uint8_t type_code = code_from_typeid(features[0]->data_type());
    // if requested, force the output format to use floats instead of doubles
    if(d_->write_float_features)
    {
      type_code = type_traits<float>::code;
    }
    ar( type_code );
    switch( type_code )
    {
      case type_traits<float>::code:
        save_features<Archive_t, float>(ar, features);
        break;
      case type_traits<double>::code:
        save_features<Archive_t, double>(ar, features);
        break;
      default:
        VITAL_THROW( vital::invalid_data, "features must be float or double");
    }
  }
  else
  {
    ar( cereal::make_size_tag( static_cast<cereal::size_type>(0) ) ); // number of elements
  }

  if( desc && desc->size() > 0 )
  {
    ar( cereal::make_size_tag( static_cast<cereal::size_type>(desc->size()) ) ); // number of elements
    uint8_t type_code = code_from_typeid(desc->at(0)->data_type());
    ar( type_code );
    switch( type_code )
    {
#define DO_CASE(T)                                       \
      case type_traits<T>::code:                         \
        save_descriptors<Archive_t, T>(ar, desc); \
        break

      DO_CASE(uint8_t);
      DO_CASE(int8_t);
      DO_CASE(uint16_t);
      DO_CASE(int16_t);
      DO_CASE(uint32_t);
      DO_CASE(int32_t);
      DO_CASE(uint64_t);
      DO_CASE(int64_t);
      DO_CASE(float);
      DO_CASE(double);
#undef DO_CASE

      default:
        VITAL_THROW( vital::invalid_data, std::string("descriptor type not supported: ")
                     + desc->at(0)->data_type().name());
    }
  }
  else
  {
    ar( cereal::make_size_tag( static_cast<cereal::size_type>(0) ) ); // number of elements
  }
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
