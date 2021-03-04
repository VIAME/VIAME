// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
