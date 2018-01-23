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
 * \brief Implementation of a thread pool
 *
 * This design is modeled after an implementation by Jakob Progsch and
 * Vaclav Zeman found here:
 *
 * https://github.com/progschj/ThreadPool
 */


#include "thread_pool.h"

#include <vital/util/thread_pool_builtin_backend.h>
#include <vital/util/thread_pool_gcd_backend.h>
#include <vital/util/thread_pool_sync_backend.h>
#include <vital/logger/logger.h>


namespace kwiver {
namespace vital {


/// Private implementation class
class thread_pool::priv
{
public:

  priv()
    : logger( kwiver::vital::get_logger( "vital.thread_pool" ) )
  {
    backend.reset( new thread_pool_builtin_backend() );
  }

  // logger handle
  logger_handle_t logger;

  // a pointer to the active backend
  std::unique_ptr<thread_pool::backend> backend;
};


/// Access the singleton instance of this class
thread_pool& thread_pool::instance()
{
  static thread_pool instance;

  return instance;
}


// Constructor
thread_pool::thread_pool()
  : d_(new priv)
{
}


/// Returns the number of worker threads
size_t thread_pool::num_threads() const
{
  return d_->backend->num_threads();;
}


/// Return the name of the active backend
const char*
thread_pool::active_backend() const
{
  return d_->backend->name();
}


/// Return the names of the available backends
std::vector<std::string>
thread_pool::available_backends()
{
  static std::vector<std::string> available_backends_list = {
#if __APPLE__
    thread_pool_gcd_backend::static_name,
#endif
    thread_pool_builtin_backend::static_name,
    thread_pool_sync_backend::static_name
  };

  return available_backends_list;
}


/// Set the backend
void thread_pool::set_backend(std::string const& backend_name)
{
#define TRY_BACKEND(T)                  \
  if(backend_name == T::static_name)    \
  {                                     \
    d_->backend.release();              \
    d_->backend.reset( new T() );       \
  }                                     \
  else

#if __APPLE__
  TRY_BACKEND( thread_pool_gcd_backend )
#endif
  TRY_BACKEND( thread_pool_builtin_backend )
  TRY_BACKEND( thread_pool_sync_backend )
  // final "else" case
  {
    LOG_ERROR( d_->logger, "Unknown thread pool backend: " << backend_name );
  }
#undef TRY_BACKEND
}


/// Enqueue a void function in the thread pool
void thread_pool::enqueue_task(std::function<void()> task)
{
  d_->backend->enqueue_task(task);
}

} }   // end namespace
