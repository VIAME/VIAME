/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "parallel.h"
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace viame
{
namespace image_kernels
{
namespace
{
// A fork inherits mutexes and thread handles but none of the other threads.
// Children use serial kernels and leave the inherited pool untouched.
long process_id ()
{
#ifdef _WIN32
  return 0;
#else
  return static_cast<long> ( getpid () );
#endif
}
thread_local bool on_worker = false;
struct pool
{
  long const owner = process_id ();
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::function<void ()>> queue;
  std::vector<std::thread> workers;
  bool stopping = false;

  pool ()
  {
    try
    {
      for ( std::size_t i = 0; i < kernel_thread_count (); ++i )
        workers.emplace_back (
            [this]
            {
              on_worker = true;
              for ( ;; )
              {
                std::function<void ()> task;
                {
                  std::unique_lock<std::mutex> lock ( mutex );
                  ready.wait ( lock, [this] { return stopping || !queue.empty (); } );
                  if ( stopping && queue.empty () )
                  {
                    return;
                  }
                  task = std::move ( queue.front () );
                  queue.pop_front ();
                }
                task ();
              }
            } );
    }
    catch ( ... )
    {
      shutdown ();
      throw;
    }
  }
  void shutdown ()
  {
    {
      std::lock_guard<std::mutex> lock ( mutex );
      stopping = true;
    }
    ready.notify_all ();
    for ( auto &worker : workers )
    {
      worker.join ();
    }
  }
  ~pool () { shutdown (); }
};
} // namespace
namespace
{
/// Zero means "whatever the environment asked for". See
/// `set_kernel_thread_count`.
std::atomic<std::size_t> thread_budget{ 0 };
} // namespace
void set_kernel_thread_count ( std::size_t count )
{
  thread_budget.store ( count, std::memory_order_relaxed );
}
std::size_t kernel_thread_count ()
{
  static auto const owner = process_id ();
  if ( process_id () != owner )
  {
    return 1;
  }
  static auto const count = []
  {
    auto const hardware = std::max ( 1u, std::thread::hardware_concurrency () );
    if ( auto const *value = std::getenv ( "VIAME_NUM_THREADS" ) )
    {
      char *end = nullptr;
      auto const wanted = std::strtol ( value, &end, 10 );
      if ( end != value && *end == '\0' && wanted > 0 )
        return std::min<std::size_t> ( wanted, hardware );
    }
    return std::min ( 4u, hardware ) + std::size_t{ 0 };
  }();
  auto const wanted = thread_budget.load ( std::memory_order_relaxed );
  return wanted ? std::min ( wanted, count ) : count;
}
void parallel_rows ( std::size_t begin, std::size_t end, std::size_t grain,
                     std::function<void ( std::size_t, std::size_t )> const &work )
{
  if ( end <= begin )
  {
    return;
  }
  auto const jobs =
      std::min ( kernel_thread_count (),
                 1 + ( end - begin - 1 ) / std::max<std::size_t> ( 1, grain ) );
  if ( jobs == 1 || on_worker )
  {
    work ( begin, end );
    return;
  }
  auto const destroy_pool = [] ( pool *p )
  {
    if ( p->owner == process_id () )
    {
      delete p;
    }
  };
  static std::unique_ptr<pool, decltype ( destroy_pool )> instance ( new pool,
                                                                     destroy_pool );
  auto &executor = *instance;
  struct completion
  {
    std::mutex mutex;
    std::condition_variable ready;
    std::size_t remaining = 0;
    std::exception_ptr error;
  };
  auto done = std::make_shared<completion> ();
  // Assemble tasks before publishing any, so allocation failure cannot leave
  // queued callbacks referencing a caller that has already unwound.
  std::deque<std::function<void ()>> tasks;
  auto callback =
      std::make_shared<std::function<void ( std::size_t, std::size_t )>> ( work );
  for ( std::size_t i = 0; i < jobs; ++i )
  {
    auto const first = begin + ( end - begin ) * i / jobs;
    auto const last = begin + ( end - begin ) * ( i + 1 ) / jobs;
    tasks.emplace_back (
        [done, callback, first, last]
        {
          std::exception_ptr error;
          try
          {
            ( *callback ) ( first, last );
          }
          catch ( ... )
          {
            error = std::current_exception ();
          }
          std::lock_guard<std::mutex> lock ( done->mutex );
          if ( error && !done->error )
          {
            done->error = error;
          }
          if ( --done->remaining == 0 )
          {
            done->ready.notify_one ();
          }
        } );
  }
  {
    std::lock_guard<std::mutex> lock ( executor.mutex );
    // Publish under the queue lock, then wait even if only some tasks could
    // be queued. Callbacks must finish before caller-owned buffers go away.
    std::lock_guard<std::mutex> finished ( done->mutex );
    try
    {
      for ( auto &task : tasks )
      {
        executor.queue.push_back ( std::move ( task ) );
        ++done->remaining;
      }
    }
    catch ( ... )
    {
      done->error = std::current_exception ();
    }
  }
  executor.ready.notify_all ();
  std::unique_lock<std::mutex> lock ( done->mutex );
  done->ready.wait ( lock, [&] { return done->remaining == 0; } );
  if ( done->error )
  {
    std::rethrow_exception ( done->error );
  }
}
} // namespace image_kernels
} // namespace viame
