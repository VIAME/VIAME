/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_exceptions.h>

#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/scheduler.h>
#include <vistk/pipeline/scheduler_exception.h>

#include <vistk/python/util/python_gil.h>

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/override.hpp>
#include <boost/python/pure_virtual.hpp>

/**
 * \file scheduler.cxx
 *
 * \brief Python bindings for \link vistk::scheduler\endlink.
 */

using namespace boost::python;

class wrap_scheduler
  : public vistk::scheduler
  , public wrapper<vistk::scheduler>
{
  public:
    wrap_scheduler(vistk::pipeline_t const& pipe, vistk::config_t const& config);
    ~wrap_scheduler();

    void _start();
    void _wait();
    void _pause();
    void _resume();
    void _stop();

    vistk::pipeline_t _pipeline() const;

    override get_pure_override(char const* name);
};

BOOST_PYTHON_MODULE(scheduler)
{
  class_<wrap_scheduler, boost::noncopyable>("PythonScheduler"
    , "The base class for Python schedulers."
    , no_init)
    .def(init<vistk::pipeline_t, vistk::config_t>())
    .def("start", &vistk::scheduler::start
      , "Start the execution of the pipeline.")
    .def("wait", &vistk::scheduler::wait
      , "Wait until the pipeline execution is complete.")
    .def("pause", &vistk::scheduler::pause
      , "Pause execution.")
    .def("resume", &vistk::scheduler::resume
      , "Resume execution.")
    .def("stop", &vistk::scheduler::stop
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
::wrap_scheduler(vistk::pipeline_t const& pipe, vistk::config_t const& config)
  : vistk::scheduler(pipe, config)
{
}

wrap_scheduler
::~wrap_scheduler()
{
}

void
wrap_scheduler
::_start()
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(get_pure_override("_start")())
}

void
wrap_scheduler
::_wait()
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(get_pure_override("_wait")())
}

void
wrap_scheduler
::_pause()
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(get_pure_override("_pause")())
}

void
wrap_scheduler
::_resume()
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(get_pure_override("_resume")())
}

void
wrap_scheduler
::_stop()
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(get_pure_override("_stop")())
}

vistk::pipeline_t
wrap_scheduler
::_pipeline() const
{
  return pipeline();
}

override
wrap_scheduler
::get_pure_override(char const* name)
{
  override const o = get_override(name);

  if (!o)
  {
    std::ostringstream sstr;
    sstr << name << " is not implemented";
    throw std::runtime_error(sstr.str().c_str());
  }

  return o;
}
