// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_TEST_INTERFACE_H
#define VITAL_TEST_INTERFACE_H

#include <string>
#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/config/config_helpers.txx>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

namespace kwiver::vital {

class test_interface : public pluggable
{
public:
  PLUGGABLE_INTERFACE( test_interface )

  virtual std::string test() = 0;

  virtual void set_configuration_internal(
    [[maybe_unused]] config_block_sptr cb ) {}
  virtual void set_configuration( [[maybe_unused]] config_block_sptr cb ) {}

  virtual config_block_sptr
  get_configuration() const
  {
    config_block_sptr cb = config_block::empty_config();
    return cb;
  }

protected:
  virtual void initialize() {}
};
typedef std::shared_ptr< test_interface > test_interface_sptr;

// ----------------------------------------------------------------------------

/**
 * This impl shows use of more explicit generator macros.
 */
class test_impl_simple : public test_interface
{
public:
  PLUGGABLE_IMPL_BASIC(
    test_impl_simple,
    "This is a simple implementation with no parameters."
  )

  PLUGGABLE_VARIABLES()

  PLUGGABLE_CONSTRUCTOR( test_impl_simple )

  PLUGGABLE_STATIC_FROM_CONFIG(
    test_impl_simple,
  )

  PLUGGABLE_STATIC_GET_DEFAULT()

  std::string
  test() override
  {
    return "simple impl";
  }
};

// ----------------------------------------------------------------------------

class test_impl_parameterized : public test_interface
{
public:
//  // Define an x-macro to enumerate parameter structures.
// #define PARAM_SET() \
//   PARAM( a, int, "some integer" ), \
//   PARAM_DEFAULT( b, std::string, "some string", "foo" )
//
//   // Expand x-macro contents into the thing that wants parameter structures.
//   PLUGGABLE_IMPL(
//     test_impl_parameterized,
//     "This is a test plugin using nesting",
//     PARAM_SET()
//     )

  // or just do it inline if we're only providing it once anyway...
  PLUGGABLE_IMPL(
    test_impl_parameterized,
    "This is a test plugin using nesting",
    // parameters
    PARAM( a, int, "some integer" ),
    PARAM_DEFAULT( b, std::string, "some string", "foo" )
  )

  std::string
  test() override
  {
    std::stringstream ss;
    ss << "class with parameters like " << c_a << " and '" + c_b + "'.";
    return ss.str();
  }
};

} // end namespace kwiver::vital

#endif /* VITAL_TEST_INTERFACE_H */
