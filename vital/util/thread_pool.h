// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
#include <vital/util/vital_util_export.h>

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace kwiver {
namespace vital {

/// A thread pool class to distribute tasks across a fixed pool of threads
/**
 *  This class provides an interface for an application wide thread pool that
 *  uses a fixed number of threads, each of which executes tasks from a task
 *  queue.  The scheduling and load balancing is dependent on the chosen
 *  backend implementation.  Several backends are available depending on
 *  platform and availability of third-party packages.  When tasks
 *  are added to the queue the enqueue function returns an std::future
 *  referring to the future value to be computed by the task.
 *
 *  Here is an example of how to use it.
 *  \code

    // functions to call, lambdas here, but could also be declared functions
    auto my_func1 = [] (int x) { return static_cast<double>(x) + 2.0; }
    auto my_func2 = [] (float x, unsigned y) { return x + y; }

    // enqueue function calls (non-blocking)
    std::future<double> val1 = thread_pool::instance().enqueue( my_func1, 10 );
    std::future<float> val2 = thread_pool::instance().enqueue( my_func2, 2.1, 3 );

    // get the results (blocks until each task is running)
    std::cout << "results " << val1.get() << ", " << val2.get() << std::endl;

 *  \endcode
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
  size_t num_threads() const;

  /// Return the name of the active backend
  const char* active_backend() const;

  /// Return the names of the available backends
  static std::vector<std::string> available_backends();

  /// Set the backend
  /**
   * Destroys the current backend and replaces it with a new
   * one of the specified type.  The \p backend_name must match
   * one of the names provided by available_backend().
   */
  void set_backend(std::string const& backend_name);

  /// Enqueue an arbitrary function as a task to run
  template<class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>;

  /// A base class for thread pool backend implementations
  class VITAL_UTIL_EXPORT backend
    : private kwiver::vital::noncopyable
  {
  public:
    /// Constructor
    backend()  VITAL_DEFAULT_CTOR

    /// Destructor
    virtual ~backend() = default;

    /// Returns the number of worker threads
    virtual size_t num_threads() const = 0;

    /// Returns the name of this backend
    virtual const char* name() const = 0;

    /// Enqueue a void() task
    virtual void enqueue_task(std::function<void()> func) = 0;
  };

private:

   /// Constructor - private for signleton
  thread_pool();

  /// Destructor
  ~thread_pool() = default;

  /// Enqueue a void function in the thread pool
  void enqueue_task(std::function<void()> task);

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
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

  // add the task to the queue using a lambda function to ignore return type
  this->enqueue_task([task](){ (*task)(); });

  return res;
}

} }   // end namespace

#endif
