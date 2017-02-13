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

#include <test_common.h>

#include <vector>

#include <vital/util/thread_pool.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


IMPLEMENT_TEST(number_of_threads)
{
  using namespace kwiver::vital;

  TEST_EQUAL( "Num thread == num CPU cores",
              thread_pool::instance().num_threads(),
              std::thread::hardware_concurrency());
}


void run_jobs_impl()
{
  using namespace kwiver::vital;

  // futures to collect
  std::vector<std::future<double> > futures;
  // lamda function to run in threads
  auto func = [] (unsigned i)
  {
    double x = static_cast<double>(i);
    return x*x;
  };

  // enqueue all the jobs
  for( unsigned i=0; i<100; i++ )
  {
    futures.push_back( thread_pool::instance().enqueue( func, i ) );
  }

  // collect all the results
  for( unsigned i=0; i<100; i++ )
  {
    TEST_EQUAL( "threaded value "+std::to_string(i), futures[i].get(), func(i) );
  }

}


IMPLEMENT_TEST(run_jobs)
{
  using namespace kwiver::vital;
  VITAL_FOREACH( std::string const& backend,
                 thread_pool::instance().available_backends() )
  {
    std::cout << "Testing with thread pool backend: " << backend << std::endl;
    thread_pool::instance().set_backend(backend);
    TEST_EQUAL( "Backend set correctly",
                thread_pool::instance().active_backend(), backend );
    run_jobs_impl();
  }
}
