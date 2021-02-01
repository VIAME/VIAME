// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 *
 * \brief Supplemental macro definitions for test cases
 */

#ifndef KWIVER_TEST_TEST_GTEST_H_
#define KWIVER_TEST_TEST_GTEST_H_

#include <gtest/gtest.h>

#define TEST_LOAD_PLUGINS() \
  class _test_plugin_helper : public ::testing::EmptyTestEventListener \
  { \
  public: \
    virtual void OnTestProgramStart(::testing::UnitTest const&) override \
    { kwiver::vital::plugin_manager::instance().load_all_plugins(); } \
  }; \
  ::testing::UnitTest::GetInstance()->listeners().Append( \
    new _test_plugin_helper )

// ----------------------------------------------------------------------------
/** @brief Consume a required command line argument.
 *
 * @param idx Index of the required argument.
 * @param var Name of global variable into which the argument will be stored.
 *
 * @note The variable must be copy-assignable, and it must be possible to
 * construct an instance of the variable's type from a \c char*. Use
 * GET_ARG_EX if the latter condition does not hold.
 */
#define GET_ARG(idx, var) \
  GET_ARG_EX(idx, var, decltype(var))

// ----------------------------------------------------------------------------
/** @brief Consume a required command line argument.
 *
 * @param idx Index of the required argument.
 * @param var Name of global variable into which the argument will be stored.
 * @param conv A functor which accepts a \c char* and returns an object that is
 *   copy-assignable to \p var.
 */
#define GET_ARG_EX(idx, var, conv)  \
  do                                \
  {                                 \
    using namespace testing;        \
    if (!GTEST_FLAG(list_tests))    \
    {                               \
      if (argc <= (idx))            \
      {                             \
        EXPECT_GT(argc, (idx))      \
          << "Required argument "   \
          << (idx) << " missing";   \
        return EXIT_FAILURE;        \
      }                             \
      var = conv(argv[idx]);        \
    }                               \
  } while (false)

// ----------------------------------------------------------------------------
/** @brief Declare a used command line argument.
 *
 * @param var Name of variable by which the command line argument will be
 *   accessed.
 *
 * @note The global variable must be named <code>g_ ## var</code>.
 */
#define TEST_ARG(var) \
  public: decltype(g_ ## var) const& var = g_ ## var

namespace kwiver {
namespace testing {

// ----------------------------------------------------------------------------
template <typename V, typename L, typename U>
::testing::AssertionResult
is_in_inclusive_range(
  char const* expr_value, char const* expr_lower, char const* expr_upper,
  V value, L lower, U upper )
{
  using namespace ::testing;

  if ( value >= lower && value <= upper )
  {
    return AssertionSuccess();
  }

  return AssertionFailure()
    << "Expected: ("
    << expr_lower << ") ≤ ("
    << expr_value << ") ≤ ("
    << expr_upper << "), where\n"
    << expr_lower << " evaluates to " << PrintToString(lower) << ",\n"
    << expr_upper << " evaluates to " << PrintToString(upper) << ", and\n"
    << expr_value << " evaluates to " << PrintToString(value) << ".";
}

#define EXPECT_WITHIN( lower, value, upper ) \
  EXPECT_PRED_FORMAT3( ::kwiver::testing::is_in_inclusive_range, \
                       value, lower, upper )

#define ASSERT_WITHIN( lower, value, upper ) \
  ASSERT_PRED_FORMAT3( ::kwiver::testing::is_in_inclusive_range, \
                       value, lower, upper )

}
}

#endif
