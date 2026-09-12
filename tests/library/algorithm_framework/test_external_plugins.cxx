/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief `VIAME_PLUGIN_PATH`, which is the one way in for code VIAME was
/// not built with.
///
/// P8-T03 deleted the directory scan; this is what replaced it, and it is
/// the only `dlopen` left in the tree. What the tests hold it to is the part
/// a plugin author depends on: that a named library is loaded and its
/// factory is usable, and that a bad entry costs them that entry and nothing
/// else.

#include <viame/algorithm_framework/plugin/plugin_loader.h>
#include <viame/algorithm_framework/registry/external_plugins.h>
#include <viame/algorithm_framework/test_interface/say.h>

#include <gtest/gtest.h>

#include <viame/algorithm_framework/config/config_block.h>

#include <cstdlib>
#include <memory>
#include <string>

namespace kv = kwiver::vital;

namespace {

// The plugin built beside this test; see `external_plugin.cxx`.
std::string const plugin_library( VIAME_TEST_EXTERNAL_PLUGIN );

// ----------------------------------------------------------------------------
void
set_plugin_path( std::string const& value )
{
  if( value.empty() )
  {
    unsetenv( viame::plugin_path_variable );
  }
  else
  {
    setenv( viame::plugin_path_variable, value.c_str(), 1 );
  }
}

// ----------------------------------------------------------------------------
/// Is there a factory for `say` under this name?
bool
has_say( kv::plugin_loader const& loader, std::string const& name )
{
  for( auto const& fact : loader.get_factories< kv::say >() )
  {
    std::string found;

    if( fact->get_attribute( kv::plugin_factory::PLUGIN_NAME, found ) &&
        found == name )
    {
      return true;
    }
  }

  return false;
}

// ----------------------------------------------------------------------------
class external_plugins : public ::testing::Test
{
protected:
  void
  TearDown() override
  {
    set_plugin_path( {} );
  }

  kv::plugin_loader loader;
};

} // namespace

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
// Unset is the case every VIAME install is in, so it is the one that has to
// cost nothing and load nothing.
TEST_F( external_plugins, an_unset_path_registers_nothing )
{
  set_plugin_path( {} );

  EXPECT_TRUE( viame::register_external_plugins( loader ).empty() );
  EXPECT_FALSE( has_say( loader, "external" ) );
}

// ----------------------------------------------------------------------------
TEST_F( external_plugins, an_empty_path_registers_nothing )
{
  setenv( viame::plugin_path_variable, "", 1 );

  EXPECT_TRUE( viame::register_external_plugins( loader ).empty() );
}

// ----------------------------------------------------------------------------
TEST_F( external_plugins, a_named_library_registers )
{
  set_plugin_path( plugin_library );

  auto const loaded = viame::register_external_plugins( loader );

  ASSERT_EQ( 1u, loaded.size() );
  EXPECT_EQ( plugin_library, loaded.front() );
  EXPECT_TRUE( has_say( loader, "external" ) );
}

// ----------------------------------------------------------------------------
// The factory has to work, not merely be listed: the code behind it lives in
// a library this process was not linked against.
TEST_F( external_plugins, the_registered_factory_runs )
{
  set_plugin_path( plugin_library );
  viame::register_external_plugins( loader );

  kv::plugin_factory_handle_t external;

  for( auto const& fact : loader.get_factories< kv::say >() )
  {
    std::string name;
    fact->get_attribute( kv::plugin_factory::PLUGIN_NAME, name );

    if( name == "external" )
    {
      external = fact;
    }
  }

  ASSERT_TRUE( external );

  auto const instance = std::dynamic_pointer_cast< kv::say >(
    external->from_config( kv::config_block::empty_config() ) );

  ASSERT_TRUE( instance );
  EXPECT_EQ( "I came from outside the build", instance->says() );
}

// ----------------------------------------------------------------------------
// The attribute says which library a factory came from, and for these that
// is the path the user named -- which is the only way to tell, from a
// registry dump, what is VIAME's and what is not.
TEST_F( external_plugins, the_factory_records_where_it_came_from )
{
  set_plugin_path( plugin_library );
  viame::register_external_plugins( loader );

  for( auto const& fact : loader.get_factories< kv::say >() )
  {
    std::string name;
    fact->get_attribute( kv::plugin_factory::PLUGIN_NAME, name );

    if( name != "external" )
    {
      continue;
    }

    std::string origin;
    EXPECT_TRUE(
      fact->get_attribute( kv::plugin_factory::PLUGIN_ORIGIN_LIBRARY,
        origin ) );
    EXPECT_EQ( plugin_library, origin );
  }
}

// ----------------------------------------------------------------------------
// A list, so the separator has to work, and the empty entries a trailing or
// doubled separator leaves behind have to be ignored rather than opened --
// an empty path means the current directory.
TEST_F( external_plugins, empty_entries_are_skipped )
{
  set_plugin_path( ":" + plugin_library + "::" );

  auto const loaded = viame::register_external_plugins( loader );

  ASSERT_EQ( 1u, loaded.size() );
  EXPECT_EQ( plugin_library, loaded.front() );
}

// ----------------------------------------------------------------------------
// Naming the same library twice is a user error that costs nothing: the
// second registration finds the factory already there and keeps the first.
TEST_F( external_plugins, naming_a_library_twice_is_harmless )
{
  set_plugin_path( plugin_library + ":" + plugin_library );

  EXPECT_EQ( 2u, viame::register_external_plugins( loader ).size() );
  EXPECT_TRUE( has_say( loader, "external" ) );
}

// ----------------------------------------------------------------------------
TEST_F( external_plugins, a_missing_file_is_skipped )
{
  set_plugin_path( "/nonexistent/libnothing.so" );

  EXPECT_TRUE( viame::register_external_plugins( loader ).empty() );
}

// ----------------------------------------------------------------------------
// A real library that is not a plugin. This is the case the directory scan
// used to get wrong by construction, because everything in the directory
// looked like a candidate.
TEST_F( external_plugins, a_library_without_the_entry_point_is_skipped )
{
  set_plugin_path( VIAME_TEST_NOT_A_PLUGIN );

  EXPECT_TRUE( viame::register_external_plugins( loader ).empty() );
}

// ----------------------------------------------------------------------------
// One bad entry costs the caller that entry, not the list.
TEST_F( external_plugins, a_bad_entry_does_not_stop_the_good_ones )
{
  set_plugin_path( "/nonexistent/libnothing.so:" + plugin_library );

  auto const loaded = viame::register_external_plugins( loader );

  ASSERT_EQ( 1u, loaded.size() );
  EXPECT_EQ( plugin_library, loaded.front() );
  EXPECT_TRUE( has_say( loader, "external" ) );
}
