/*ckwg +29
 * Copyright 2017-2025 by Kitware, Inc.
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
 * \brief Process to add keypoints to detections from oriented bounding box
 */

#include "ocv_add_keypoints_from_poly.h"
#include "ocv_stereo_utils.h"

#include <vital/types/detected_object_set.h>

#include <sprokit/processes/kwiver_type_traits.h>

namespace kv = kwiver::vital;

namespace viame
{

// =============================================================================
// Private implementation class
class ocv_add_keypoints_from_poly::priv
{
public:
  explicit priv( ocv_add_keypoints_from_poly* parent );
  ~priv();

  ocv_add_keypoints_from_poly* parent;
};

// -----------------------------------------------------------------------------
ocv_add_keypoints_from_poly::priv
::priv( ocv_add_keypoints_from_poly* ptr )
  : parent( ptr )
{
}

// -----------------------------------------------------------------------------
ocv_add_keypoints_from_poly::priv
::~priv()
{
}

// =============================================================================
ocv_add_keypoints_from_poly
::ocv_add_keypoints_from_poly( kv::config_block_sptr const& config )
  : process( config ),
    d( new ocv_add_keypoints_from_poly::priv( this ) )
{
  make_ports();
  make_config();
}

ocv_add_keypoints_from_poly
::~ocv_add_keypoints_from_poly()
{
}

// -----------------------------------------------------------------------------
void
ocv_add_keypoints_from_poly
::make_ports()
{
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // Input port
  declare_input_port_using_trait( detected_object_set, required );

  // Output port
  declare_output_port_using_trait( detected_object_set, required );
}

// -----------------------------------------------------------------------------
void
ocv_add_keypoints_from_poly
::make_config()
{
  // No configuration options needed for this simple process
}

// -----------------------------------------------------------------------------
void
ocv_add_keypoints_from_poly
::_configure()
{
  // Nothing to configure
}

// -----------------------------------------------------------------------------
void
ocv_add_keypoints_from_poly
::_step()
{
  // Grab input
  auto detection_set = grab_from_port_using_trait( detected_object_set );

  // Process each detection
  std::vector<kv::detected_object_sptr> output_dets;

  for( const auto& det : *detection_set )
  {
    // Only add keypoints if the detection has a mask
    if( det->mask() )
    {
      add_keypoints_from_box( det );
    }

    output_dets.push_back( det );
  }

  // Create output set
  auto output_set = std::make_shared<kv::detected_object_set>( output_dets );

  // Push output
  push_to_port_using_trait( detected_object_set, output_set );
}

} // end namespace viame
