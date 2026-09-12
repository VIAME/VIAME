/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief A plugin shaped like one VIAME was not built with.
///
/// It is built here rather than written out in the test because the thing
/// under test is what happens to a real shared library: that it can be
/// opened, that the entry point can be found in it by name, and that a
/// factory registered from inside it is usable from outside. None of that
/// survives being faked.

#include <viame/algorithm_framework/plugin/plugin_loader.h>
#include <viame/algorithm_framework/test_interface/say.h>

namespace kwiver::vital {

class external_say : public say
{
public:
  external_say() = default;
  ~external_say() override = default;

  std::string
  says() override
  {
    return "I came from outside the build";
  }

  static pluggable_sptr
  from_config( config_block_sptr const /* cb */ )
  {
    return std::make_shared< external_say >();
  }

  static void
  get_default_config( config_block& /* cb */ )
  {}
};

} // namespace kwiver::vital

namespace kv = kwiver::vital;

extern "C"
__attribute__( ( visibility( "default" ) ) )
void
viame_register_plugin( kwiver::vital::plugin_loader& loader )
{
  loader.add_factory< kv::say, kv::external_say >( "external" );
}
