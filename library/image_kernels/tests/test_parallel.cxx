/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <image_kernels/parallel.h>
#include <gtest/gtest.h>
#include <atomic>
#include <future>
#include <stdexcept>
#include <vector>
namespace ik = viame::image_kernels;
int main ( int argc, char **argv )
{
  ::testing::InitGoogleTest ( &argc, argv );
  return RUN_ALL_TESTS ();
}
TEST ( parallel, nested_and_concurrent_callers )
{
  std::atomic<int> count{ 0 };
  std::vector<std::future<void>> callers;
  for ( int i = 0; i < 8; ++i )
    callers.emplace_back (
        std::async ( std::launch::async,
                     [&]
                     {
                       ik::parallel_rows (
                           0, 16, 1,
                           [&] ( std::size_t first, std::size_t last )
                           {
                             for ( auto row = first; row < last; ++row )
                               ik::parallel_rows (
                                   0, 7, 1, [&] ( std::size_t a, std::size_t b )
                                   { count.fetch_add ( static_cast<int> ( b - a ) ); } );
                           } );
                     } ) );
  for ( auto &caller : callers )
  {
    caller.get ();
  }
  EXPECT_EQ ( count, 8 * 16 * 7 );
}
TEST ( parallel, exceptions_propagate_and_workers_survive )
{
  EXPECT_THROW ( ik::parallel_rows ( 0, 16, 1, [] ( std::size_t, std::size_t )
                                     { throw std::runtime_error ( "worker failure" ); } ),
                 std::runtime_error );
  std::atomic<int> count{ 0 };
  ik::parallel_rows ( 0, 13, 1,
                      [&] ( std::size_t a, std::size_t b ) { count += b - a; } );
  EXPECT_EQ ( count, 13 );
}

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#include <cstdlib>
TEST ( parallel, forked_child_does_not_wait_for_inherited_workers )
{
  ik::parallel_rows ( 0, 16, 1, [] ( std::size_t, std::size_t ) {} );
  auto child = fork ();
  ASSERT_GE ( child, 0 );
  if ( child == 0 )
  {
    alarm ( 5 );
    int count = 0;
    ik::parallel_rows ( 0, 13, 1,
                        [&] ( std::size_t a, std::size_t b ) { count += b - a; } );
    // exit also exercises the inherited singleton's destructor.
    std::exit ( count == 13 && ik::kernel_thread_count () == 1 ? 0 : 1 );
  }
  int status = 0;
  ASSERT_EQ ( waitpid ( child, &status, 0 ), child );
  ASSERT_TRUE ( WIFEXITED ( status ) );
  EXPECT_EQ ( WEXITSTATUS ( status ), 0 );
}
#endif
