/*ckwg +29
 * Copyright 2017, 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *  to endorse or promote products derived from this software without specific
 *  prior written permission.
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
