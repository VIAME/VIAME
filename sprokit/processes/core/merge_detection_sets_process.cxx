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

#include "merge_detection_sets_process.h"

#include <kwiver_type_traits.h>

#include <vital/types/detected_object_set.h>

namespace kwiver
{

create_port_trait( detected_object_set1, detected_object_set,
   "A first detected object set");

create_port_trait( detected_object_set2, detected_object_set,
   "A second detected object set");

create_port_trait( detected_object_set_out, detected_object_set,
   "A detected object set to output");

merge_detection_sets_process
::merge_detection_sets_process( vital::config_block_sptr const& config )
  : process( config )
{
  // Attach our logger name to process logger
  attach_logger( vital::get_logger( name() ) );

  make_ports();
  make_config();
}

merge_detection_sets_process
::~merge_detection_sets_process()
{
}

void merge_detection_sets_process
::_configure()
{
}

void
merge_detection_sets_process
::_step()
{
  vital::detected_object_set_sptr set1 = grab_from_port_using_trait(detected_object_set1);
  vital::detected_object_set_sptr set2 = grab_from_port_using_trait(detected_object_set2);

  vital::detected_object_set_sptr set_out(new vital::detected_object_set());
  set_out->add(set1);
  set_out->add(set2);

  push_to_port_using_trait(detected_object_set_out, set_out);
}

void
merge_detection_sets_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait(detected_object_set1, required);
  declare_input_port_using_trait(detected_object_set2, optional);

  // -- output --
  declare_output_port_using_trait(detected_object_set_out, required);
}

void
merge_detection_sets_process
::make_config()
{

}

} // end namespace
