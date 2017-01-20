/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Interface and implementation of a thread pool
 *
 * This design is modeled after an implementation by Jakob Progsch and
 * Vaclav Zeman found here:
 *
 * https://github.com/progschj/ThreadPool
 */

#ifndef KWIVER_VITAL_THREAD_POOL_H_
#define KWIVER_VITAL_THREAD_POOL_H_

#include <vital/noncopyable.h>
#include <vital/vital_foreach.h>
#include <vital/util/vital_util_export.h>

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace kwiver {
namespace vital {

/// A thread pool class to distribute tasks across a fixed pool of threads
/**
 *  This class spawns a fixed number of thread, each of which runs in an
 *  endless loop pulling tasks off of a queue and executing them.  When tasks
 *  are added to the queue the enqueue function returns an std::future
 *  referring to the future value to be computed by the task.
 */
class VITAL_UTIL_EXPORT thread_pool
  : private kwiver::vital::noncopyable
{
public:
  /// Access the singleton instance of this class
  /**
   * \returns The reference to the singleton instance.
   */
  static thread_pool& instance();

  /// Returns the number of worker threads
  size_t size() const;

  /// Enqueue an arbitrary function as a task to run
  template<class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>;

private:

   /// Constructor - create a thread pool with some number of threads
  thread_pool(size_t num_threads = std::thread::hardware_concurrency());

  /// Destructor - joins all threads
  ~thread_pool();

  /// This function is executed in each thread to endlessly process tasks
  void thread_worker_loop();

  /// The task queue
  std::queue< std::function<void()> > tasks;

  /// The collection of threads in the pool
  std::vector<std::thread> workers;

  /// Mutex to synchronize access to the queue
  std::mutex queue_mutex;

  /// Condition variable to allow threads to wait for tasks
  std::condition_variable condition;

  /// Flag to indicate that the processing loop should terminate
  bool stop;
};


/// Enqueue an arbitrary function as a task to run
template<class F, class... Args>
auto thread_pool::enqueue(F&& f, Args&&... args)
  -> std::future<typename std::result_of<F(Args...)>::type>
{
  // get the return type of the function to be run
  using return_type = typename std::result_of<F(Args...)>::type;

  // package up the task
  auto task = std::make_shared< std::packaged_task<return_type()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

  // get a future to the function result to return to the caller
  std::future<return_type> res = task->get_future();

  // add the task to the queue as long as it still running
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    // don't allow enqueueing after stopping the pool
    if(stop)
    {
      throw std::runtime_error("enqueue on stopped thread_pool");
    }

    // add the task to the queue using a lambda function to ignore return type
    tasks.emplace([task](){ (*task)(); });
  }
  // notify one worker to start processing
  condition.notify_one();

  return res;
}

} }   // end namespace

#endif // KWIVER_VITAL_THREAD_POOL_H_
