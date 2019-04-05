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
 * \brief Write VIAME homographies list
 */

#include "write_homography_list_process.h"

#include <vital/vital_types.h>
#include <vital/types/homography.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <fstream>
#include <iostream>
#include <exception>


namespace viame
{

namespace core
{

create_config_trait( file_name, std::string, "homographies.txt",
  "Filename for writing homographies into" );
create_config_trait( no_homography_string, std::string, "[no-match]",
  "String to print out to the output file if there is no match" );

create_port_trait( source_file_name, file_name, "Source file name" );
create_port_trait( dest_file_name, file_name, "Destination file name" );
create_port_trait( homography, homography, "Homography mapping Source to Dest" );

//------------------------------------------------------------------------------
// Private implementation class
class write_homography_list_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_name;
  std::string m_no_homography_string;

  // Internal variables
  std::ofstream m_writer;
};

// =============================================================================

write_homography_list_process
::write_homography_list_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new write_homography_list_process::priv() )
{
  make_ports();
  make_config();
}


write_homography_list_process
::~write_homography_list_process()
{
  if( d->m_writer.is_open() )
  {
    d->m_writer.close();
  }
}


// -----------------------------------------------------------------------------
void
write_homography_list_process
::_configure()
{
  d->m_file_name = config_value_using_trait( file_name );
  d->m_no_homography_string = config_value_using_trait( no_homography_string );

  if( d->m_writer.is_open() )
  {
    d->m_writer.close();
  }

  d->m_writer.open( d->m_file_name, std::ofstream::out );

  if( !d->m_writer.is_open() )
  {
    throw std::runtime_error( "Unable to open " + d->m_file_name );
  }
}


// -----------------------------------------------------------------------------
void
write_homography_list_process
::_step()
{
  std::string source_file_name;
  std::string dest_file_name;

  kwiver::vital::homography_sptr homog;

  source_file_name = grab_from_port_using_trait( source_file_name );
  dest_file_name = grab_from_port_using_trait( dest_file_name );
  homog = grab_from_port_using_trait( homography );

  if( !source_file_name.empty() )
  {
    d->m_writer << source_file_name << std::endl;
  }
  if( !dest_file_name.empty() )
  {
    d->m_writer << dest_file_name << std::endl;
  }

  if( homog )
  {
    d->m_writer << *homog << std::endl;
  }
  else
  {
    d->m_writer << d->m_no_homography_string << std::endl;
  }

  d->m_writer << std::endl;
}


// -----------------------------------------------------------------------------
void
write_homography_list_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( source_file_name, required );
  declare_input_port_using_trait( dest_file_name, required );
  declare_input_port_using_trait( homography, required );
}


// -----------------------------------------------------------------------------
void
write_homography_list_process
::make_config()
{
  declare_config_using_trait( file_name );
  declare_config_using_trait( no_homography_string );
}


// =============================================================================
write_homography_list_process::priv
::priv()
{
}


write_homography_list_process::priv
::~priv()
{
}


} // end namespace core

} // end namespace viame
