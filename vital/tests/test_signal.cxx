// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core signal/context tests
 */

#include <vital/signal.h>

#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <thread>

namespace kv = kwiver::vital;

// ----------------------------------------------------------------------------
int
main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(signal, basic)
{
  kv::signal< int > signal;
  kv::context ctx;

  auto result = int{ 0 };

  signal.connect( &ctx, [ & ]( int value ){ result = value; } );

  signal( 42 );
  EXPECT_EQ( 42, result );

  signal( 17 );
  EXPECT_EQ( 17, result );
}

// ----------------------------------------------------------------------------
TEST(signal, multiple_slots)
{
  kv::signal< int > signal;

  auto result1 = int{ 0 };
  auto result2 = int{ 0 };
  auto result3 = int{ 1 };

  auto ctx1 = std::unique_ptr< kv::context >{ new kv::context };
  signal.connect( ctx1.get(), [ & ]( int value ){ result1 = value; } );

  auto ctx2 = std::unique_ptr< kv::context >{ new kv::context };
  signal.connect( ctx2.get(), [ & ]( int value ){ result2 += value; } );

  auto ctx3 = std::unique_ptr< kv::context >{ new kv::context };
  signal.connect( ctx3.get(), [ & ]( int value ){ result3 *= value; } );

  signal( 42 );
  EXPECT_EQ( 42, result1 );
  EXPECT_EQ( 42, result2 );
  EXPECT_EQ( 42, result3 );

  signal( 17 );
  EXPECT_EQ( 17, result1 );
  EXPECT_EQ( 59, result2 );
  EXPECT_EQ( 714, result3 );

  ctx3.reset();

  signal( 42 );
  EXPECT_EQ( 42, result1 );
  EXPECT_EQ( 101, result2 );
  EXPECT_EQ( 714, result3 );
}

// ----------------------------------------------------------------------------
TEST(signal, races)
{
  // This test merits some explanation. The goal of this test is to verify that
  // a context can be destroyed without racing the emission or destruction of a
  // signal. The first is relatively easy to force, but for the second, we can
  // only rely on running the test many times and hoping for lucky scheduling.
  //
  // To execute the test, we create a context in a separate thread, notify that
  // the thread is running, and wait for the signal to be raised. Meanwhile, on
  // the original/main thread, we wait for the notification, then raise the
  // signal. The slot fires in the main thread, notifies that it is running,
  // and immediately goes to sleep so that the signal will be "busy". Back on
  // the second thread, upon receiving the notification, we try to destroy the
  // context, which will block because the signal is "busy".
  //
  // Finally, in the original/main thread, once the signal finishes executing
  // the slot, we immediately destroy the signal. Depending on timing, this may
  // or may not happen before the context is destroyed. Running the test
  // repeatedly should exercise both possibilities.

  auto signal = std::unique_ptr< kv::signal<> >{ new kv::signal<> };
  std::atomic< kv::context* > ctxp{ nullptr };
  std::atomic< bool > cond{ false };

  auto thread = std::thread{
    [ & ]{
      auto ctx = std::unique_ptr< kv::context >{ new kv::context };

      signal->connect(
        ctx.get(),
        [ & ]{
          // Notify that the signal is executing
          cond = false;

          // Go to sleep so that the secondary thread will start to tear down
          // the context; note that this slot is executing in the original/main
          // thread, since that is where the signal is raised
          std::this_thread::sleep_for( std::chrono::milliseconds{ 250 } );
        });

      // Notify that we are ready for the signal to be raised, then wait
      cond = true;
      while ( cond.load() )
      {
        std::this_thread::yield();
      }

      // Destroy our context
      ctx.reset();
    }};

  // Wait until thread is executing
  while ( !cond.load() )
  {
    std::this_thread::yield();
  }

  // Raise the signal
  ( *signal )();

  // Destroy the signal, hopefully while the context is still being destroyed
  signal.reset();

  // Wait for thread to terminate
  thread.join();
}
