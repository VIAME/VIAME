/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * \brief Accept vector of doubles from descriptor.
 */

#include "accept_descriptor.h"
#include "io_mgr.h"

#include <vital/vital_types.h>

#include <kwiver_util/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver {

// should be promoted to project level include
create_port_trait( d_vector, double_vector, "Vector of doubles from descriptor" );

// config items
//          None for now

//----------------------------------------------------------------
// Private implementation class
class accept_descriptor::priv
{
public:
  priv();
  ~priv();


  // empty for now
};


// ================================================================

accept_descriptor
::accept_descriptor( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new accept_descriptor::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) );

  make_ports();
  make_config();
}


accept_descriptor
::~accept_descriptor()
{
}


// ----------------------------------------------------------------
void
accept_descriptor
::_configure()
{

}


// ----------------------------------------------------------------
void
accept_descriptor
::_step()
{
  kwiver::vital::double_vector_sptr vect = grab_from_port_using_trait( d_vector );

  kwiver::io_mgr::Instance()->SetDescriptor( vect );
}


// ----------------------------------------------------------------
void
accept_descriptor
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( d_vector, required );
}


// ----------------------------------------------------------------
void
accept_descriptor
::make_config()
{
}


// ================================================================
accept_descriptor::priv
::priv()
{
}


accept_descriptor::priv
::~priv()
{
}

} // end namespace
