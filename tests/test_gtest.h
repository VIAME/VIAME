/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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

#endif
