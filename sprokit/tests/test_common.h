// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_TEST_TEST_COMMON_H
#define SPROKIT_TEST_TEST_COMMON_H

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

#define DECLARE_TEST_MAP()                                    \
  namespace                                                   \
  {                                                           \
    typedef boost::function<void TEST_ARGS> test_function_t;  \
    typedef std::map<testname_t, test_function_t> test_map_t; \
  }                                                           \
  test_map_t __all_tests;                                     \
  struct __add_test                                           \
  {                                                           \
    __add_test(testname_t const& name,                        \
               test_function_t const& func)                   \
    {                                                         \
      __all_tests[name] = func;                               \
    }                                                         \
  }                                                           \

#define TEST_PROPERTY(property, value, ...)

#define IMPLEMENT_TEST(testname)                       \
  static void                                          \
  test_##testname TEST_ARGS;                           \
  static __add_test const                              \
    __add_test_##testname(#testname, test_##testname); \
  void                                                 \
  test_##testname TEST_ARGS

#define CHECK_ARGS(numargs)     \
  do                            \
  {                             \
    if (argc != (numargs + 1))  \
    {                           \
      TEST_ERROR("Expected "    \
                 #numargs       \
                 " arguments"); \
                                \
      return EXIT_FAILURE;      \
    }                           \
  } while (false)

#define RUN_TEST(testname, ...)                 \
  do                                            \
  {                                             \
    test_map_t::const_iterator const i =        \
      __all_tests.find(testname);               \
                                                \
    if (i == __all_tests.end())                 \
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

#endif // SPROKIT_TEST_TEST_COMMON_H
