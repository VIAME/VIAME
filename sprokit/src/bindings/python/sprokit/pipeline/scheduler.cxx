/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/scheduler.h>

#include <sprokit/python/util/python_exceptions.h>
#include <sprokit/python/util/python_gil.h>

#if WIN32
#pragma warning (push)
#pragma warning (disable : 4244)
#endif
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/override.hpp>
#include <boost/python/pure_virtual.hpp>
#if WIN32
#pragma warning (pop)
#endif

/**
 * \file scheduler.cxx
 *
 * \brief Python bindings for \link sprokit::scheduler\endlink.
 */

using namespace boost::python;

class wrap_scheduler
  : public sprokit::scheduler
  , public wrapper<sprokit::scheduler>
{
  public:
    wrap_scheduler(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config);
    ~wrap_scheduler();

    void _start();
    void _wait();
    void _pause();
    void _resume();
    void _stop();

    sprokit::pipeline_t _pipeline() const;

    override get_pure_override(char const* name) const;
};

BOOST_PYTHON_MODULE(scheduler)
{
  class_<wrap_scheduler, boost::noncopyable>("PythonScheduler"
    , "The base class for Python schedulers."
    , no_init)
    .def(init<sprokit::pipeline_t, kwiver::vital::config_block_sptr>())
    .def("start", &sprokit::scheduler::start
      , "Start the execution of the pipeline.")
    .def("wait", &sprokit::scheduler::wait
      , "Wait until the pipeline execution is complete.")
    .def("pause", &sprokit::scheduler::pause
      , "Pause execution.")
    .def("resume", &sprokit::scheduler::resume
      , "Resume execution.")
    .def("stop", &sprokit::scheduler::stop
      , "Stop the execution of the pipeline.")
    .def("pipeline", &wrap_scheduler::_pipeline
      , "The pipeline the scheduler is to run.")
    .def("_start", pure_virtual(&wrap_scheduler::_start)
      , "Implementation of starting the pipeline.")
    .def("_wait", pure_virtual(&wrap_scheduler::_wait)
      , "Implementation of waiting until execution is complete.")
    .def("_pause", pure_virtual(&wrap_scheduler::_pause)
      , "Implementation of pausing execution.")
    .def("_resume", pure_virtual(&wrap_scheduler::_resume)
      , "Implementation of resuming execution.")
    .def("_stop", pure_virtual(&wrap_scheduler::_stop)
      , "Implementation of stopping the pipeline.")
  ;
}

wrap_scheduler
::wrap_scheduler(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config)
  : sprokit::scheduler(pipe, config)
{
}

wrap_scheduler
::~wrap_scheduler()
{
  shutdown();
}

void
wrap_scheduler
::_start()
{
  sprokit::python::python_gil const gil;

  (void)gil;

  SPROKIT_PYTHON_HANDLE_EXCEPTION(get_pure_override("_start")())
}

void
wrap_scheduler
::_wait()
{
  sprokit::python::python_gil const gil;

  (void)gil;

  SPROKIT_PYTHON_HANDLE_EXCEPTION(get_pure_override("_wait")())
}

void
wrap_scheduler
::_pause()
{
  sprokit::python::python_gil const gil;

  (void)gil;

  SPROKIT_PYTHON_HANDLE_EXCEPTION(get_pure_override("_pause")())
}

void
wrap_scheduler
::_resume()
{
  sprokit::python::python_gil const gil;

  (void)gil;

  SPROKIT_PYTHON_HANDLE_EXCEPTION(get_pure_override("_resume")())
}

void
wrap_scheduler
::_stop()
{
  sprokit::python::python_gil const gil;

  (void)gil;

  SPROKIT_PYTHON_HANDLE_EXCEPTION(get_pure_override("_stop")())
}

sprokit::pipeline_t
wrap_scheduler
::_pipeline() const
{
  return pipeline();
}

override
wrap_scheduler
::get_pure_override(char const* method) const
{
  override const o = get_override(method);

  if (!o)
  {
    std::ostringstream sstr;

    sstr << method << " is not implemented";

    throw std::runtime_error(sstr.str().c_str());
  }

  return o;
}
