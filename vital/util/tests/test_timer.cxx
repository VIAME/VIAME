// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test util timer class
 */

#include <vital/util/cpu_timer.h>
#include <vital/util/wall_timer.h>
#include <kwiversys/SystemTools.hxx>

#include <gtest/gtest.h>

#include <iostream>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

namespace  {

// Which Fibonacci number to compute for tests; must be large enough that the
// computation takes non-negligible time
constexpr unsigned fib_n = 37;

// Expected value of fibonacci( fib_n )
constexpr long fib_n_value = 24157817;

// Millisecond sleep duration; must be larger than clock resolution
constexpr unsigned sleep_duration = 100;

// ----------------------------------------------------------------------------
// Classic bad Fibonacci implementation
long fibonacci( unsigned n )
{
    if ( n < 2 ) return n;
    return fibonacci( n - 1 ) + fibonacci( n - 2 );
}

} // end namespace

// ----------------------------------------------------------------------------
TEST(timer, cpu_timer)
{
  kwiver::vital::cpu_timer timer;
  double t1, t2;
  long fib;

  // As of the current implementation, CPU timers should always be available
  ASSERT_TRUE( timer.timer_available() );

  EXPECT_FALSE( timer.is_active() );
  EXPECT_EQ( 0, timer.elapsed() );

  timer.start();
  EXPECT_TRUE( timer.is_active() );

  fib = fibonacci( fib_n );
  EXPECT_NE( 0, t1 = timer.elapsed() );

  // Displaying the result of this call is important; otherwise it can be
  // optimized out of generated code resulting in no elapsed time
  std::cout << "fib(" << fib_n << ") = " << fib << std::endl;

  // Computing a different value is important; otherwise this call could be
  // optimized out
  fib = fibonacci( fib_n + 1 );
  EXPECT_NE( 0, t2 = timer.elapsed() );
  EXPECT_LT( t1, t2 );

  std::cout << "fib(" << fib_n + 1 << ") = " << fib << std::endl;

  timer.stop();
  t1 = timer.elapsed();

  fib = fibonacci( fib_n - 1 );
  std::cout << "fib(" << fib_n - 1 << ") = " << fib << std::endl;

  EXPECT_EQ( t1, timer.elapsed() );
}

// ----------------------------------------------------------------------------
TEST(timer, wall_timer)
{
  kwiver::vital::wall_timer timer;
  double t1, t2;

  // As of the current implementation, wall timers should always be available
  ASSERT_TRUE( timer.timer_available() );

  EXPECT_FALSE( timer.is_active() );
  EXPECT_EQ( 0, timer.elapsed() );

  timer.start();
  EXPECT_TRUE( timer.is_active() );

  kwiversys::SystemTools::Delay( sleep_duration );
  EXPECT_NE( 0, t1 = timer.elapsed() );

  kwiversys::SystemTools::Delay( sleep_duration );
  EXPECT_NE( 0, t2 = timer.elapsed() );
  EXPECT_LT( t1, t2 );

  timer.stop();
  t1 = timer.elapsed();

  kwiversys::SystemTools::Delay( sleep_duration );
  EXPECT_EQ( t1, timer.elapsed() );
}

// ----------------------------------------------------------------------------
TEST(timer, scoped_cpu_timer)
{
  // As of the current implementation, CPU timers should always be available
  ASSERT_TRUE( kwiver::vital::cpu_timer{}.timer_available() );

  kwiver::vital::scoped_cpu_timer timer( "cpu_fib_test" );
  EXPECT_EQ( fib_n_value, fibonacci( fib_n ) );
}

// ----------------------------------------------------------------------------
TEST(timer, scoped_wall_timer)
{
  // As of the current implementation, wall timers should always be available
  ASSERT_TRUE( kwiver::vital::wall_timer{}.timer_available() );

  kwiver::vital::scoped_wall_timer timer( "wall_fib_test" );
  EXPECT_EQ( fib_n_value, fibonacci( fib_n ) );
}
