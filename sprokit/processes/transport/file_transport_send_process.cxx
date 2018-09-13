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

#include "file_transport_send_process.h"

#include <sprokit/pipeline/process_exception.h>

#include <kwiver_type_traits.h>

#include <fstream>


namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( file_name, std::string, "serialized_data.dat",
                     "Name of file where serialized messages are written. ");


//----------------------------------------------------------------
// Private implementation class
class file_transport_send_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_name;

  std::ofstream m_output_file;

}; // end priv class

// ================================================================

file_transport_send_process
::file_transport_send_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new file_transport_send_process::priv )
{
  make_ports();
  make_config();
}


file_transport_send_process
::~file_transport_send_process()
{
}


// ----------------------------------------------------------------
void file_transport_send_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get process config entries
  d->m_file_name = config_value_using_trait( file_name );

  d->m_output_file.open( d->m_file_name );
  if ( ! d->m_output_file )
  {
    throw sprokit::invalid_configuration_exception( name(),
                     "Unable to open output file." );
  }
}


// ----------------------------------------------------------------
void file_transport_send_process
::_step()
{
  auto mess = grab_from_port_using_trait( serialized_message );

  scoped_step_instrumentation();

  // We know that the message is a pointer to a std::string
  d->m_output_file << *mess;
}


// ----------------------------------------------------------------
void file_transport_send_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_using_trait( serialized_message, required );
}


// ----------------------------------------------------------------
void file_transport_send_process
::make_config()
{
  declare_config_using_trait( file_name );
}


// ================================================================
file_transport_send_process::priv
::priv()
  : m_file_name( "transport_file.dat" )
{
}


file_transport_send_process::priv
::~priv()
{
}





} // end namespace
