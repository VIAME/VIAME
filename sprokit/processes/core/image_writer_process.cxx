/*ckwg +29
 * Copyright 2016-2017, 2020 by Kitware, Inc.
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

#include "image_writer_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/image.h>
#include <vital/types/timestamp.h>
#include <vital/algo/image_io.h>
#include <vital/exceptions.h>
#include <vital/util/string.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/datum.h>

#include <kwiversys/SystemTools.hxx>

#include <vector>
#include <stdint.h>
#include <fstream>

// -- DEBUG
#if defined DEBUG
#include <arrows/algorithms/ocv/image_container.h>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif

namespace algo = kwiver::vital::algo;

namespace kwiver
{

// (config-key, value-type, default-value, description )
create_config_trait( file_name_template, std::string, "",
  "Template for generating output file names. The template is interpreted "
  "as a printf format with one format specifier to convert an integer "
  "increasing image number. The image file type is determined by the file "
  "extension and the concrete writer selected. If an image name over-ride "
  "is provided over a pipeline, only the extension in the name is used.");

create_config_trait( replace_filename_strings, std::string , "",
  "An optional comma-seperated list of pairs, corresponding to filename "
  "components we wish to replace. For example, if this is color,grey "
  "all instances of the word color in the filename will be replaced "
  "with grey." );

create_algorithm_name_config_trait( image_writer );

// -----------------------------------------------------------------------------
// Private implementation class
class image_writer_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_template;

  // Number for current image.
  kwiver::vital::frame_id_t m_frame_number;

  // Optional pipeline input parameter
  std::string m_filename_override;

  // processing classes
  algo::image_io_sptr m_image_writer;

  // replace strings
  std::vector< std::string > m_find_strings;
  std::vector< std::string > m_replace_strings;
}; // end priv class


// =============================================================================

image_writer_process
::image_writer_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new image_writer_process::priv )
{
  make_ports();
  make_config();
}


image_writer_process
::~image_writer_process()
{
}


// -----------------------------------------------------------------------------
void image_writer_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get process config entries
  d->m_file_template = config_value_using_trait( file_name_template );

  // Get algo config entries
  kwiver::vital::config_block_sptr algo_config = get_config();

  algo::image_io::set_nested_algo_configuration_using_trait(
    image_writer,
    algo_config,
    d->m_image_writer);
  if ( !d->m_image_writer )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Unable to create image_writer." );
  }

  // instantiate image reader and converter based on config type
  if( ! algo::image_io::check_nested_algo_configuration_using_trait(
        image_writer,
        algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Configuration check failed." );
  }


  // Identify find-replace strings
  std::string full_string = config_value_using_trait( replace_filename_strings );

  std::vector< std::string > parsed_string;
  std::stringstream iss( full_string );
  std::string tmp;

  while( getline( iss, tmp, ',' ) )
  {
    parsed_string.push_back( tmp );
  }

  if( parsed_string.size() % 2 == 1 )
  {
    throw sprokit::invalid_configuration_exception( name(),
      "Length of replace string vector must be even." );
  }

  for( unsigned i = 0; i < parsed_string.size(); i+=2 )
  {
    d->m_find_strings.push_back( parsed_string[i] );
    d->m_replace_strings.push_back( parsed_string[i+1] );
  }
}


// -----------------------------------------------------------------------------
void image_writer_process
::_step()
{
  if( has_input_port_edge_using_trait( timestamp ) )
  {
    kwiver::vital::timestamp frame_time;
    frame_time = grab_from_port_using_trait( timestamp );

    if( frame_time.has_valid_frame() )
    {
      kwiver::vital::frame_id_t next_frame;
      next_frame = frame_time.get_frame();

      if( next_frame <= d->m_frame_number )
      {
        ++d->m_frame_number;
        LOG_WARN( logger(), "Frame number from input timestamp ("
                  << next_frame
                  << ") is not greater than last frame number. "
                  << "Adjusting frame number to "
                  << d->m_frame_number );
      }
    }
    else
    {
      // timestamp does not have valid frame number
      ++d->m_frame_number;
    }
  }
  else
  {
    // timestamp port not connected.
    ++d->m_frame_number;
  }

  vital::image_container_sptr input = grab_from_port_using_trait( image );

  std::string a_file;

  if( has_input_port_edge_using_trait( image_file_name ) )
  {
    a_file = grab_from_port_using_trait( image_file_name );

    if( !d->m_file_template.empty() && d->m_file_template[0] == '.' )
    {
      a_file = a_file.substr( 0, a_file.find_last_of( "." ) ) + d->m_file_template;
    }
  }
  else
  {
    a_file = kwiver::vital::string_format( d->m_file_template, d->m_frame_number );
  }

  for( unsigned i = 0; i < d->m_find_strings.size(); ++i )
  {
    size_t start_pos = 0;

    while( ( start_pos = a_file.find( d->m_find_strings[i], start_pos ) ) != std::string::npos )
    {
      a_file.replace( start_pos, d->m_find_strings[i].length(), d->m_replace_strings[i] );
      start_pos += d->m_replace_strings[i].length();
    }
  }

  if( input )
  {
    scoped_step_instrumentation();
    LOG_DEBUG( logger(), "Writing image to file \"" << a_file << "\"" );

    d->m_image_writer->save( a_file, input );
  }

  bool output_flag = ( input && input->size() > 0 );

  push_to_port_using_trait( success_flag, output_flag );
  push_to_port_using_trait( image_file_name, ( output_flag ? a_file : "" ) );
}


// -----------------------------------------------------------------------------
void image_writer_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_using_trait( image, required );

  declare_input_port_using_trait( image_file_name, optional,
    "Name of the image file to write. If not specified, the pattern config "
    "parameter will be used instead." );

  declare_input_port_using_trait( timestamp, optional,
    "Image timestamp, optional. The frame number from this timestamp is used to "
    "number the output files. If the timestamp is not connected or not valid, "
    "the output files are sequentially numbered from 1." );

  declare_output_port_using_trait( image_file_name, optional,
    "Name of the image file written to." );

  declare_output_port_using_trait( success_flag, optional,
    "Flag indicating the image write was successful" );
}


// -----------------------------------------------------------------------------
void image_writer_process
::make_config()
{
  declare_config_using_trait( file_name_template );
  declare_config_using_trait( image_writer );
  declare_config_using_trait( replace_filename_strings );
}


// =============================================================================
image_writer_process::priv
::priv()
  : m_frame_number(0)
{
}


image_writer_process::priv
::~priv()
{
}

} // end namespace
