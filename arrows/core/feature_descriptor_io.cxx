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

#include <vital/logger/logger.h>


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


/// Implementation specific load functionality.
void
feature_descriptor_io
::load_(std::string const& filename,
        vital::feature_set_sptr& feat,
        vital::descriptor_set_sptr& desc) const
{
}


/// Implementation specific save functionality.
void
feature_descriptor_io
::save_(std::string const& filename,
        vital::feature_set_sptr feat,
        vital::descriptor_set_sptr desc) const
{
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
