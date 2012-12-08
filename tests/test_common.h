/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_TEST_TEST_COMMON_H
#define VISTK_TEST_TEST_COMMON_H

#include <boost/function.hpp>

#include <exception>
#include <iostream>
#include <map>
#include <string>

#include <cstdlib>

typedef std::string testname_t;

#define TEST_ERROR(msg)                         \
  do                                            \
  {                                             \
    std::cerr << "Error: " << msg << std::endl; \
  } while (false)

#define EXPECT_EXCEPTION(ex, code, action)  \
  do                                        \
  {                                         \
    bool got_exception = false;             \
                                            \
    try                                     \
    {                                       \
      code;                                 \
    }                                       \
    catch (ex const& e)                     \
    {                                       \
      got_exception = true;                 \
                                            \
      std::cerr << "Expected exception: "   \
                << e.what()                 \
                << std::endl;               \
    }                                       \
    catch (std::exception const& e)         \
    {                                       \
      TEST_ERROR("Unexpected exception: "   \
                 << e.what());              \
                                            \
      got_exception = true;                 \
    }                                       \
    catch (...)                             \
    {                                       \
      TEST_ERROR("Non-standard exception"); \
                                            \
      got_exception = true;                 \
    }                                       \
                                            \
    if (!got_exception)                     \
    {                                       \
      TEST_ERROR("Did not get "             \
                 "expected exception when " \
                 << action);                \
    }                                       \
  } while (false)

#define CHECK_ARGS(numargs)        \
  do                               \
  {                                \
    if (argc != (numargs + 1))     \
    {                              \
      TEST_ERROR("Expected "       \
                 << numargs        \
                 << " arguments"); \
                                   \
      return EXIT_FAILURE;         \
    }                              \
  } while (false)

#define DECLARE_TEST(testname) \
  static void test_##testname TEST_ARGS

#define ADD_TEST(tests, testname) \
  tests[#testname] = test_##testname

#define IMPLEMENT_TEST(testname) \
  void                           \
  test_##testname TEST_ARGS

#define DECLARE_TEST_MAP(tests)                             \
  typedef boost::function<void TEST_ARGS> test_function_t;  \
  typedef std::map<testname_t, test_function_t> test_map_t; \
                                                            \
  test_map_t tests

#define RUN_TEST(tests, testname, ...)          \
  do                                            \
  {                                             \
    test_map_t::const_iterator const i =        \
      tests.find(testname);                     \
                                                \
    if (i == tests.end())                       \
    {                                           \
      TEST_ERROR("Unknown test: " << testname); \
                                                \
      return EXIT_FAILURE;                      \
    }                                           \
                                                \
    test_function_t const& func = i->second;    \
                                                \
    try                                         \
    {                                           \
      func(__VA_ARGS__);                        \
    }                                           \
    catch (std::exception const& e)             \
    {                                           \
      TEST_ERROR("Unexpected exception: "       \
                 << e.what());                  \
                                                \
      return EXIT_FAILURE;                      \
    }                                           \
                                                \
    return EXIT_SUCCESS;                        \
  } while (false)

#endif // VISTK_TEST_TEST_COMMON_H
