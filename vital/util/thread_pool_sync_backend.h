// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of a thread pool backend that runs jobs synchronously
 */

#ifndef KWIVER_VITAL_THREAD_POOL_SYNC_BACKEND_H_
#define KWIVER_VITAL_THREAD_POOL_SYNC_BACKEND_H_

#include <vital/util/thread_pool.h>

namespace kwiver {
namespace vital {

/// A thread pool backend that runs jobs synchronously (e.g. no threads)
class thread_pool_sync_backend
  : public thread_pool::backend
{
public:
  /// The name of this backend
  static constexpr const char* static_name = "Sync";

  /// Enqueue a void() task
  void enqueue_task(std::function<void()> func)
  {
    func();
  }

  /// Returns the number of worker threads
  size_t num_threads() const
  {
    return 0;
  }

  /// Returns the name of this backend
  virtual const char* name() const
  {
    return static_name;
  }
};

} }   // end namespace

#endif
