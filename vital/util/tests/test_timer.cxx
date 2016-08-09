/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief test util timer class
 */
#include <test_common.h>

#include <vital/util/cpu_timer.h>
#include <vital/util/wall_timer.h>
#include <kwiversys/SystemTools.hxx>

#include <iostream>

typedef kwiversys::SystemTools ST;

#define TEST_ARGS ()

DECLARE_TEST_MAP();

namespace  {

// classic bad Fibonacci implementation.
long fibonacci(unsigned n)
{
    if (n < 2) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

} // end namespace

// ------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(cpu_timer_test)
{
  kwiver::vital::cpu_timer timer;

  if ( ! timer.timer_available() )
  {
    std::cout << "Skipping tests, CPU Timer support not available\n";
    return;
  }

  TEST_EQUAL( "CPU Timers supported", timer.timer_available(), true );
  TEST_EQUAL( "CPU Timer not active", timer.is_active(), false );
  TEST_EQUAL( "Inactive CPU Timer interval", timer.elapsed(), 0 );

  timer.start();
  TEST_EQUAL( "CPU Timer active", timer.is_active(), true );

  long fib = fibonacci(40);

  double t1 = timer.elapsed();
  bool value = (t1 != 0);
  TEST_EQUAL( "CPU Timer 1 not zero", value, true );
  // Displaying the result of this call is important.  Otherwise
  // it can be optimized out of generated code resulting in no elapsed
  // time.
  std::cout << "fib(40) = " << fib << std::endl;

  // Computing a different value is important.  Otherwise this
  // call could be optimized out.
  fib = fibonacci(41);

  double t2 = timer.elapsed();
  value = (t2 != 0);
  TEST_EQUAL( "CPU Timer 2 not zero", value, true );
  std::cout << "fib(41) = " << fib << std::endl;

  value = (t1 == t2);
  TEST_EQUAL( "CPU Timers not the same", value, false );

  timer.stop();
  t1 = timer.elapsed();

  fib = fibonacci(39);
  std::cout << "fib(39) = " << fib << std::endl;

  t2 = timer.elapsed();

  TEST_EQUAL( "Stopped CPU Timer does not change", t1, t2 );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(wall_timer_test)
{
  kwiver::vital::wall_timer timer;

  if ( ! timer.timer_available() )
  {
    std::cout << "Skipping tests, timer support not available\n";
    return;
  }

  TEST_EQUAL( "Wall Timers supported", timer.timer_available(), true );
  TEST_EQUAL( "Wall Timer not active", timer.is_active(), false );
  TEST_EQUAL( "Inactive Wall Timer interval", timer.elapsed(), 0 );

  timer.start();
  TEST_EQUAL( "Wall Timer active", timer.is_active(), true );

  ST::Delay(1000);

  double t1 = timer.elapsed();
  bool value = (t1 != 0);
  TEST_EQUAL( "Wall Timer 1 not zero", value, true );

  ST::Delay(1000);

  double t2 = timer.elapsed();
  value = (t2 != 0);
  TEST_EQUAL( "Wall Timer 2 not zero", value, true );

  value = (t1 == t2);
  TEST_EQUAL( "Wall Timers not the same", value, false );

  timer.stop();
  t1 = timer.elapsed();
  ST::Delay(1000);
  t2 = timer.elapsed();

  TEST_EQUAL( "Stopped Wall Timer does not change", t1, t2 );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(fib_test)
{
  kwiver::vital::cpu_timer a_timer;
  if ( ! a_timer.timer_available() )
  {
    std::cout << "Skipping tests, timer support not available\n";
    std::cout << "Timer support not available\n";

  }

  long fib(0);
  {
    kwiver::vital::scoped_cpu_timer timer( "cpu_fib_test" );
    fib = fibonacci(42);
    std::cout << "  f(42) = " << fib << std::endl;
  }

  {
    kwiver::vital::scoped_wall_timer timer( "wall_fib_test" );
    fib = fibonacci(42);
    std::cout << "  f(42) = " << fib << std::endl;
  }

  TEST_EQUAL( "Expected Fibonacci number", fib, 267914296 );
}
