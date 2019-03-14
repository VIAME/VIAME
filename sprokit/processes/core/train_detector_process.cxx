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

#include "train_detector_process.h"

#include <vital/algo/train_detector.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver {

create_config_trait( trainer, std::string, "",
  "Algorithm configuration subblock." );

// -----------------------------------------------------------------------------
// Private implementation class
class train_detector_process::priv
{
public:
  priv();
  ~priv();

  vital::algo::train_detector_sptr m_trainer;
}; // end priv class


// =============================================================================
train_detector_process
::train_detector_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new train_detector_process::priv )
{
  set_data_checking_level( check_none );

  make_ports();
  make_config();
}


train_detector_process
::~train_detector_process()
{
}


// -----------------------------------------------------------------------------
void
train_detector_process
::_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  if( !vital::algo::train_detector::check_nested_algo_configuration(
        "trainer", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Configuration check failed." );
  }

  vital::algo::train_detector::set_nested_algo_configuration(
    "trainer", algo_config, d->m_trainer );

  if( !d->m_trainer )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Unable to create trainer" );
  }
}


// -----------------------------------------------------------------------------
void
train_detector_process
::_step()
{
  auto port_info = peek_at_port_using_trait( image );

  if( port_info.datum->type() == sprokit::datum::complete )
  {
    // Perform training
    LOG_INFO( logger(), "Beginning training procedure" );

    d->m_trainer->update_model();

    // Complete process
    grab_edge_datum_using_trait( detected_object_set );
    mark_process_as_complete();

    const sprokit::datum_t dat = sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( object_track_set, dat );
    return;
  }

  vital::image_container_sptr image = grab_from_port_using_trait( image );
  vital::detected_object_set_sptr gt = grab_from_port_using_trait( detected_object_set );

  {
    scoped_step_instrumentation();

    d->m_trainer->add_data_from_memory(
      vital::category_hierarchy_sptr(),
      std::vector< kwiver::vital::image_container_sptr >( 1, image ),
      std::vector< kwiver::vital::detected_object_set_sptr >( 1, gt ) );
  }

  push_to_port_using_trait( object_track_set,
    std::make_shared< kwiver::vital::object_track_set >() );
}


// -----------------------------------------------------------------------------
void
train_detector_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( object_track_set, optional );
}


// -----------------------------------------------------------------------------
void
train_detector_process
::make_config()
{
  declare_config_using_trait( trainer );
}


// =============================================================================
train_detector_process::priv
::priv()
{
}


train_detector_process::priv
::~priv()
{
}

} // end namespace kwiver
