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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief test Vital thread pool class
 */

#include <vital/util/thread_pool.h>

#include <gtest/gtest.h>

#include <vector>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(thread_pool, number_of_threads)
{
  EXPECT_EQ( std::thread::hardware_concurrency(),
             thread_pool::instance().num_threads() );
}

// ----------------------------------------------------------------------------
class thread_pool_backend : public ::testing::TestWithParam<std::string>
{
};

// ----------------------------------------------------------------------------
TEST_P(thread_pool_backend, run_jobs)
{
  thread_pool::instance().set_backend( GetParam() );
  EXPECT_EQ( GetParam(), thread_pool::instance().active_backend() );

  // futures to collect
  std::vector<std::future<double> > futures;
  // lamda function to run in threads
  auto func = []( unsigned i )
  {
    double x = static_cast<double>( i );
    return x * x;
  };

  // enqueue all the jobs
  for ( unsigned i = 0; i < 100; ++i )
  {
    futures.push_back( thread_pool::instance().enqueue( func, i ) );
  }

  // collect all the results
  for ( unsigned i = 0; i < 100; ++i )
  {
    SCOPED_TRACE( "For thread " + std::to_string( i ) );
    EXPECT_EQ( func( i ), futures[i].get() );
  }
}

// ----------------------------------------------------------------------------
INSTANTIATE_TEST_CASE_P(
  ,
  thread_pool_backend,
  ::testing::ValuesIn( thread_pool::available_backends() ) );
