/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_registry.h>

#include <sprokit/config.h>

#include <boost/make_shared.hpp>

using namespace sprokit;

class SPROKIT_NO_EXPORT test_process
  : public sprokit::process
{
  public:
    test_process(config_t const& config);
    ~test_process();
};

test_process
::test_process(config_t const& config)
  : process(config)
{
}

test_process
::~test_process()
{
}

extern "C"
{

SPROKIT_EXPORT void register_processes();

}

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("test_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("test", "A test process", create_process<test_process>);

  registry->mark_module_as_loaded(module_name);
}
