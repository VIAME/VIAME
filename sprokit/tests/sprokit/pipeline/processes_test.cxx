// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_factory.h>

#include "processes_test_export.h"

using namespace sprokit;

class PROCESSES_TEST_NO_EXPORT test_process
  : public sprokit::process
{
  public:
  PLUGIN_INFO( "test",
               "A test process" );

    test_process(kwiver::vital::config_block_sptr const& config);
    ~test_process();
};

test_process
::test_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
{
}

test_process
::~test_process()
{
}

// ------------------------------------------------------------------
extern "C"
PROCESSES_TEST_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  process_registrar reg( vpm, "test_processes" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_process< test_process >();

  reg.mark_module_as_loaded();
}
