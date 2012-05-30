/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/exceptions.h>

#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/schedule.h>
#include <vistk/pipeline/schedule_exception.h>

#include <vistk/python/util/python_gil.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/override.hpp>
#include <boost/python/pure_virtual.hpp>

/**
 * \file schedule.cxx
 *
 * \brief Python bindings for \link vistk::schedule\endlink.
 */

using namespace boost::python;

class wrap_schedule
  : public vistk::schedule
  , public wrapper<vistk::schedule>
{
  public:
    wrap_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipe);
    ~wrap_schedule();

    void _start();
    void _wait();
    void _stop();

    vistk::pipeline_t _pipeline() const;

    override get_pure_override(char const* name);
};

BOOST_PYTHON_MODULE(schedule)
{
  class_<wrap_schedule, boost::noncopyable>("PythonSchedule"
    , "The base class for Python schedules."
    , no_init)
    .def(init<vistk::config_t, vistk::pipeline_t>())
    .def("start", &vistk::schedule::start
      , "Start the execution of the pipeline.")
    .def("wait", &vistk::schedule::wait
      , "Wait until the pipeline execution is complete.")
    .def("stop", &vistk::schedule::stop
      , "Stop the execution of the pipeline.")
    .def("pipeline", &wrap_schedule::_pipeline
      , "The pipeline the schedule is to run.")
    .def("_start", pure_virtual(&wrap_schedule::_start)
      , "Implementation of starting the pipeline.")
    .def("_wait", pure_virtual(&wrap_schedule::_wait)
      , "Implementation of waiting until execution is complete.")
    .def("_stop", pure_virtual(&wrap_schedule::_stop)
      , "Implementation of stopping the pipeline.")
  ;
}

wrap_schedule
::wrap_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipe)
  : vistk::schedule(config, pipe)
{
}

wrap_schedule
::~wrap_schedule()
{
}

void
wrap_schedule
::_start()
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(get_pure_override("_start")())
}

void
wrap_schedule
::_wait()
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(get_pure_override("_wait")())
}

void
wrap_schedule
::_stop()
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(get_pure_override("_stop")())
}

vistk::pipeline_t
wrap_schedule
::_pipeline() const
{
  return pipeline();
}

override
wrap_schedule
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
