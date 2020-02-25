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

#include "test_proc_seq.h"
#include <sprokit/processes/kwiver_type_traits.h>

create_type_trait( number, "integer", int32_t );

create_port_trait( number, number, "number uint 32" );


namespace kwiver {

test_proc_seq
::test_proc_seq( kwiver::vital::config_block_sptr const& config )
  : process (config )
{
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_using_trait( number, required );
}


void test_proc_seq
::_configure()
{
  scoped_configure_instrumentation();

}

void test_proc_seq
::_init()
{
  scoped_init_instrumentation();

}


void test_proc_seq
::_step()
{
  scoped_step_instrumentation();

  auto input = grab_from_port_using_trait( number );
  LOG_INFO( logger(),  "Number: " << input );
}


void test_proc_seq
::_finalize()
{
  scoped_finalize_instrumentation();

}


} // end namespace
