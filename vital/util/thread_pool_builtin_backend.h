// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
