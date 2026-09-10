// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/algorithm_framework/plugin/plugin_loader.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>
#include <viame/algorithm_framework/test_interface/say.h>
#include "say_cpp_export.h"

// ----------------------------------------------------------------------------
namespace kwiver::vital {

class cpp_say_impl : public say
{
public:
  cpp_say_impl() = default;
  ~cpp_say_impl() override = default;

  std::string
  says() override
  {
    return "I am the C++ plugin";
  }

  static pluggable_sptr
  from_config( config_block_sptr const /* cb */ )
  {
    // No parameter constructor, just return new instance
    return std::make_shared< cpp_say_impl >();
  }

  static void
  get_default_config( config_block& /* cb */ )
  {
    // No constructor parameters, so nothing to set in the config block.
  }
};

class cpp_they_say : public say
{
public:
  cpp_they_say() = default;
  ~cpp_they_say() override = default;

  std::string
  says() override
  {
    return "In C++ they say " + m_speaker->says();
  }

  static pluggable_sptr
  from_config( config_block_sptr const cb )
  {
    auto ret_val = std::make_shared< cpp_they_say >();
    ret_val->m_speaker =
      implementation_factory_by_name< say >().create(
        cb->get_value( "speaker", std::string( "cpp" ) ), cb );

    return ret_val;
  }

  static void
  get_default_config( config_block& /* cb */ )
  {
    // No constructor parameters, so nothing to set in the config block.
  }

private:
  say_sptr m_speaker;
};

} // namespace kwiver::vital

// ----------------------------------------------------------------------------
namespace kv = kwiver::vital;

extern "C"
SAY_CPP_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpl )
{
  vpl.add_factory< kv::say, kv::cpp_say_impl >( "cpp" );
  vpl.add_factory< kv::say, kv::cpp_they_say >( "cpp_they" );
}
