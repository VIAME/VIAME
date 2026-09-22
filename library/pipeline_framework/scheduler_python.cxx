// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/pipeline_framework/pipeline.h>
#include <viame/pipeline_framework/scheduler.h>

#include <viame/algorithm_framework/util/python_exceptions.h>

#include "python_wrappers.cxx"

#include <pybind11/pybind11.h>
#include "python_fold.h"

/**
 * \file scheduler.cxx
 *
 * \brief Python bindings for \link viame::pipeline::scheduler\endlink.
 */

using namespace pybind11;

namespace viame {

namespace pipeline {

namespace python {

// Publisher class to access virtual methods
class wrap_scheduler
  : public ::viame::pipeline::scheduler
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
  : public ::viame::pipeline::scheduler
{
public:
  scheduler_trampoline(
    ::viame::pipeline::pipeline_t const& pipe,
    viame::config_block_sptr const& config ) : scheduler( pipe, config )
  {}
  void _start() override;
  void _wait() override;
  void _pause() override;
  void _resume() override;
  void _stop() override;
};

void scheduler_shutdown( object );

} // namespace python

} // namespace pipeline

} // namespace viame

using namespace viame::pipeline::python;

VIAME_PYTHON_MODULE( scheduler, m )
{
  class_< viame::pipeline::scheduler, scheduler_trampoline, viame::pipeline::scheduler_t >(
    m, "PythonScheduler",
    "The base class for Python schedulers." )
    .def(
    init< viame::pipeline::pipeline_t, viame::config_block_sptr >(),
    call_guard< pybind11::gil_scoped_release >() )
    .def(
      "start", &viame::pipeline::scheduler::start,
      call_guard< pybind11::gil_scoped_release >(),
      "Start the execution of the pipeline." )
    .def(
      "wait", &viame::pipeline::scheduler::wait,
      call_guard< pybind11::gil_scoped_release >(),
      "Wait until the pipeline execution is complete." )
    .def(
      "pause", &viame::pipeline::scheduler::pause,
      call_guard< pybind11::gil_scoped_release >(),
      "Pause execution." )
    .def(
      "resume", &viame::pipeline::scheduler::resume,
      call_guard< pybind11::gil_scoped_release >(),
      "Resume execution." )
    .def(
      "stop", &viame::pipeline::scheduler::stop,
      call_guard< pybind11::gil_scoped_release >(),
      "Stop the execution of the pipeline." )
    .def(
      "_start",
      static_cast< void ( viame::pipeline::scheduler::* )() >( &wrap_scheduler::_start ),
      call_guard< pybind11::gil_scoped_release >(),
      "Implementation of starting the pipeline." )
    .def(
      "_wait",
      static_cast< void ( viame::pipeline::scheduler::* )() >( &wrap_scheduler::_wait ),
      call_guard< pybind11::gil_scoped_release >(),
      "Implementation of waiting until execution is complete." )
    .def(
      "_pause",
      static_cast< void ( viame::pipeline::scheduler::* )() >( &wrap_scheduler::_pause ),
      call_guard< pybind11::gil_scoped_release >(),
      "Implementation of pausing execution." )
    .def(
      "_resume",
      static_cast< void ( viame::pipeline::scheduler::* )() >( &wrap_scheduler::_resume ),
      call_guard< pybind11::gil_scoped_release >(),
      "Implementation of resuming execution." )
    .def(
      "_stop",
      static_cast< void ( viame::pipeline::scheduler::* )() >( &wrap_scheduler::_stop ),
      call_guard< pybind11::gil_scoped_release >(),
      "Implementation of stopping the pipeline." )
    .def(
      "pipeline",
      static_cast< viame::pipeline::pipeline_t ( viame::pipeline::scheduler::* )() const >( &
                                                                            wrap_scheduler
                                                                            ::
                                                                            pipeline ),
      call_guard< pybind11::gil_scoped_release >(),
      "Scheduler pipeline." )
    .def(
      "shutdown", &scheduler_shutdown, call_guard< gil_scoped_release >(),
      "Shut down the scheduler." )
  ;
}

namespace viame {

namespace pipeline {

namespace python {

// There's some bad refcounting, so we end up with one extra
// That one extra occurs whether we're using a C++ scheduler
// or a python scheduler (where we already had to inc_ref to counteract a
// mistaken dec_ref)
void
scheduler_shutdown( object schd_obj )
{
  schd_obj.dec_ref();
}

void
scheduler_trampoline
::_start()
{
  PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _start,
  );
}

void
scheduler_trampoline
::_wait()
{
  PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _wait,
  );
}

void
scheduler_trampoline
::_pause()
{
  PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _pause,
  );
}

void
scheduler_trampoline
::_resume()
{
  PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _resume,
  );
}

void
scheduler_trampoline
::_stop()
{
  PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _stop,
  );
}

} // namespace python

} // namespace pipeline

} // namespace viame
