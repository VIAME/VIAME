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

#include <sprokit/python/util/pybind11.h>
#include <sprokit/python/util/python_exceptions.h>

#include "python_wrappers.cxx"

#include <pybind11/pybind11.h>

/**
 * \file scheduler.cxx
 *
 * \brief Python bindings for \link sprokit::scheduler\endlink.
 */

using namespace pybind11;

// Publisher class to access virtual methods
class wrap_scheduler
  : public sprokit::scheduler
{
  public:
    using scheduler::scheduler;
    using scheduler::_start;
    using scheduler::_wait;
    using scheduler::_pause;
    using scheduler::_resume;
    using scheduler::_stop;
    using scheduler::pipeline;
};

// Trampoline class to allow us to to use virtual methods
class scheduler_trampoline
  : public sprokit::scheduler
{
  public:
    scheduler_trampoline(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config) : scheduler( pipe, config ) {};
    void _start() override;
    void _wait() override;
    void _pause() override;
    void _resume() override;
    void _stop() override;
};

void scheduler_shutdown(object);

PYBIND11_MODULE(scheduler, m)
{
  class_<sprokit::scheduler, scheduler_trampoline, sprokit::scheduler_t>(m, "PythonScheduler"
    , "The base class for Python schedulers.")
    .def(init<sprokit::pipeline_t, kwiver::vital::config_block_sptr>(), call_guard<sprokit::python::gil_scoped_release>())
    .def("start", &sprokit::scheduler::start, call_guard<sprokit::python::gil_scoped_release>()
      , "Start the execution of the pipeline.")
    .def("wait", &sprokit::scheduler::wait, call_guard<sprokit::python::gil_scoped_release>()
      , "Wait until the pipeline execution is complete.")
    .def("pause", &sprokit::scheduler::pause, call_guard<sprokit::python::gil_scoped_release>()
      , "Pause execution.")
    .def("resume", &sprokit::scheduler::resume, call_guard<sprokit::python::gil_scoped_release>()
      , "Resume execution.")
    .def("stop", &sprokit::scheduler::stop, call_guard<sprokit::python::gil_scoped_release>()
      , "Stop the execution of the pipeline.")
    .def("_start", static_cast<void (sprokit::scheduler::*)()>(&wrap_scheduler::_start), call_guard<sprokit::python::gil_scoped_release>()
      , "Implementation of starting the pipeline.")
    .def("_wait", static_cast<void (sprokit::scheduler::*)()>(&wrap_scheduler::_wait), call_guard<sprokit::python::gil_scoped_release>()
      , "Implementation of waiting until execution is complete.")
    .def("_pause", static_cast<void (sprokit::scheduler::*)()>(&wrap_scheduler::_pause), call_guard<sprokit::python::gil_scoped_release>()
      , "Implementation of pausing execution.")
    .def("_resume", static_cast<void (sprokit::scheduler::*)()>(&wrap_scheduler::_resume), call_guard<sprokit::python::gil_scoped_release>()
      , "Implementation of resuming execution.")
    .def("_stop", static_cast<void (sprokit::scheduler::*)()>(&wrap_scheduler::_stop), call_guard<sprokit::python::gil_scoped_release>()
      , "Implementation of stopping the pipeline.")
    .def("pipeline", static_cast<sprokit::pipeline_t (sprokit::scheduler::*)() const>(&wrap_scheduler::pipeline), call_guard<sprokit::python::gil_scoped_release>()
      , "Scheduler pipeline.")
	.def("shutdown", &scheduler_shutdown, call_guard<gil_scoped_release>()
      , "Shut down the scheduler.")
  ;
}

// There's some bad refcounting, so we end up with one extra
// That one extra occurs whether we're using a C++ scheduler
// or a python scheduler (where we already had to inc_ref to counteract a mistaken dec_ref)
void
scheduler_shutdown(object schd_obj)
{
  schd_obj.dec_ref();
}

void
scheduler_trampoline
::_start()
{
  SPROKIT_PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _start,
  );
}

void
scheduler_trampoline
::_wait()
{
  SPROKIT_PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _wait,
  );
}

void
scheduler_trampoline
::_pause()
{
  SPROKIT_PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _pause,
  );
}

void
scheduler_trampoline
::_resume()
{
  SPROKIT_PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _resume,
  );
}

void
scheduler_trampoline
::_stop()
{
  SPROKIT_PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _stop,
  );
}
