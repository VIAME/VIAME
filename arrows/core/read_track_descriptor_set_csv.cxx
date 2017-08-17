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
 * \brief Implementation of read_track_descriptor_set_csv
 */

#include "read_track_descriptor_set_csv.h"

#include <time.h>

#include <vital/vital_foreach.h>

namespace kwiver {
namespace arrows {
namespace core {

// -------------------------------------------------------------------------------
class read_track_descriptor_set_csv::priv
{
public:
  priv( read_track_descriptor_set_csv* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "read_track_descriptor_set_csv" ) )
    , m_first( true )
    , m_frame_number( 1 )
    , m_delim( "," )
  { }

  ~priv() { }

  read_track_descriptor_set_csv* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;
  int m_frame_number;
  std::string m_delim;
};


// ===============================================================================
read_track_descriptor_set_csv
::read_track_descriptor_set_csv()
  : d( new read_track_descriptor_set_csv::priv( this ) )
{
}


read_track_descriptor_set_csv
::~read_track_descriptor_set_csv()
{
}


// -------------------------------------------------------------------------------
void
read_track_descriptor_set_csv
::set_configuration(vital::config_block_sptr config)
{
  d->m_delim = config->get_value<std::string>( "delimiter", d->m_delim );
}


// -------------------------------------------------------------------------------
bool
read_track_descriptor_set_csv
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}


// -------------------------------------------------------------------------------
bool
read_track_descriptor_set_csv
::read_set( kwiver::vital::track_descriptor_set_sptr& set )
{
  return true;
}

} } } // end namespace
