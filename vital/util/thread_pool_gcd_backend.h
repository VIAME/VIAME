// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of a thread pool backend using Apple Grand Central Dispatch
 */

#ifdef __APPLE__

#ifndef KWIVER_VITAL_THREAD_POOL_GCD_BACKEND_H_
#define KWIVER_VITAL_THREAD_POOL_GCD_BACKEND_H_

#include <vital/util/thread_pool.h>

#include <dispatch/dispatch.h>

namespace kwiver {
namespace vital {

/// A thread pool backend that uses Apple Grand Central Dispatch
class thread_pool_gcd_backend
  : public thread_pool::backend
{
public:
  /// The name of this backend
  static constexpr const char* static_name = "Apple GCD";

  /// Enqueue a void() task
  void enqueue_task(std::function<void()> func)
  {
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    std::function<void()>* f = new std::function<void()>(func);
    dispatch_async_f(queue, f,
        [] (void* ctx)
        {
          std::function<void()>* func = static_cast<std::function<void()>* >(ctx);
          (*func)();
          delete func;
        });
  }

  /// Returns the number of worker threads
  size_t num_threads() const
  {
    // GCD does not let you know how many threads it is using
    // but we would expected it to typically be the number of cores
    return std::thread::hardware_concurrency();
  }

  /// Returns the name of this backend
  virtual const char* name() const
  {
    return static_name;
  }

};

} }   // end namespace

#endif

#endif // __APPLE__
