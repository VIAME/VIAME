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
 * \brief Implementation of a simple built-in thread pool
 *
 * This design is modeled after an implementation by Jakob Progsch and
 * Vaclav Zeman found here:
 *
 * https://github.com/progschj/ThreadPool
 */

#ifndef KWIVER_VITAL_THREAD_POOL_BUILTIN_BACKEND_H_
#define KWIVER_VITAL_THREAD_POOL_BUILTIN_BACKEND_H_

#include <vital/util/thread_pool.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>


namespace kwiver {
namespace vital {

/// A simple thread pool backend using C++11 standard library features
class thread_pool_builtin_backend
  : public thread_pool::backend
{
public:
  /// Constructor
  thread_pool_builtin_backend(size_t num_threads=std::thread::hardware_concurrency())
    : stop(false)
  {
    for(size_t i=0; i<num_threads; ++i)
    {
      workers.emplace_back([this] { thread_worker_loop(); });
    }
  }

  /// Destructor
  ~thread_pool_builtin_backend()
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for(std::thread &worker : workers)
    {
      worker.join();
    }
  }

  /// The name of this backend
  static constexpr const char* static_name = "Built-in";

  /// This function is executed in each thread to endlessly process tasks
  void thread_worker_loop();

  /// Enqueue a void() task
  void enqueue_task(std::function<void()> func);

  /// Returns the number of worker threads
  size_t num_threads() const { return workers.size(); }

  /// Returns the name of this backend
  virtual const char* name() const { return static_name; }

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


/// This function is executed in each thread to endlessly process tasks
void thread_pool_builtin_backend::thread_worker_loop()
{
  // loop forever
  for(;;)
  {
    std::function<void()> task;

    {
      std::unique_lock<std::mutex> lock(this->queue_mutex);
      this->condition.wait(lock,
        [this]{ return this->stop || !this->tasks.empty(); });
      if(this->stop && this->tasks.empty())
        return;
      task = std::move(this->tasks.front());
      this->tasks.pop();
    }

    task();
  }
}


/// Enqueue a void function in the thread pool
void thread_pool_builtin_backend::enqueue_task(std::function<void()> func)
{
  // add the task to the queue as long as it still running
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    // don't allow enqueueing after stopping the pool
    if(stop)
    {
      throw std::runtime_error("enqueue on stopped thread_pool");
    }

    // add the task to the queue
    tasks.emplace(func);
  }
  // notify one worker to start processing
  condition.notify_one();
}


} }   // end namespace

#endif
