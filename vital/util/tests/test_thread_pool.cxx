// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
