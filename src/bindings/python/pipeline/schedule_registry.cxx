/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_threading.h>

#include <vistk/pipeline/schedule.h>
#include <vistk/pipeline/schedule_registry.h>
#include <vistk/pipeline/schedule_registry_exception.h>

#include <vistk/python/util/python_gil.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/wrapper.hpp>

/**
 * \file schedule_registry.cxx
 *
 * \brief Python bindings for \link vistk::schedule_registry\endlink.
 */

using namespace boost::python;

static void register_schedule(vistk::schedule_registry_t reg,
                              vistk::schedule_registry::type_t const& type,
                              vistk::schedule_registry::description_t const& desc,
                              object obj);

BOOST_PYTHON_MODULE(schedule_registry)
{
  class_<vistk::schedule_registry::type_t>("ScheduleType"
    , "The type for a type of schedule.");
  class_<vistk::schedule_registry::description_t>("ScheduleDescription"
    , "The type for a description of a schedule type.");
  class_<vistk::schedule_registry::types_t>("ScheduleTypes"
    , "A collection of schedule types.")
    .def(vector_indexing_suite<vistk::schedule_registry::types_t>())
  ;
  class_<vistk::schedule_registry::module_t>("ScheduleModule"
    , "The type for a schedule module name.");

  class_<vistk::schedule, vistk::schedule_t, boost::noncopyable>("Schedule"
    , "An abstract class which offers an interface for pipeline execution strategies."
    , no_init)
    .def("start", &vistk::schedule::start
      , "Start the execution of the pipeline.")
    .def("wait", &vistk::schedule::wait
      , "Wait until the pipeline execution is complete.")
    .def("stop", &vistk::schedule::stop
      , "Stop the execution of the pipeline.")
  ;

  class_<vistk::schedule_registry, vistk::schedule_registry_t, boost::noncopyable>("ScheduleRegistry"
    , "A registry of all known schedule types."
    , no_init)
    .def("self", &vistk::schedule_registry::self
      , "Returns an instance of the schedule registry.")
    .staticmethod("self")
    .def("register_schedule", &register_schedule
      , (arg("type"), arg("description"), arg("ctor"))
      , "Registers a function which creates a schedule of the given type.")
    .def("create_schedule", &vistk::schedule_registry::create_schedule
      , (arg("type"), arg("pipeline"), arg("config") = vistk::config::empty_config())
      , "Creates a new schedule of the given type.")
    .def("types", &vistk::schedule_registry::types
      , "A list of known schedule types.")
    .def("description", &vistk::schedule_registry::description
      , (arg("type"))
      , "The description for the given schedule type.")
    .def("is_module_loaded", &vistk::schedule_registry::is_module_loaded
      , (arg("module"))
      , "Returns True if the module has already been loaded, False otherwise.")
    .def("mark_module_as_loaded", &vistk::schedule_registry::mark_module_as_loaded
      , (arg("module"))
      , "Marks a module as loaded.")
    .def_readonly("default_type", &vistk::schedule_registry::default_type
      , "The default schedule type.")
  ;
}

class python_schedule_wrapper
  : python_threading
{
  public:
    python_schedule_wrapper(object obj);
    ~python_schedule_wrapper();

    vistk::schedule_t operator () (vistk::pipeline_t const& pipeline, vistk::config_t const& config);
  private:
    object const m_obj;
};

void
register_schedule(vistk::schedule_registry_t reg,
                  vistk::schedule_registry::type_t const& type,
                  vistk::schedule_registry::description_t const& desc,
                  object obj)
{
  vistk::python::python_gil const gil;

  (void)gil;

  python_schedule_wrapper const wrap(obj);

  reg->register_schedule(type, desc, wrap);
}

python_schedule_wrapper
::python_schedule_wrapper(object obj)
  : m_obj(obj)
{
}

python_schedule_wrapper
::~python_schedule_wrapper()
{
}

vistk::schedule_t
python_schedule_wrapper
::operator () (vistk::pipeline_t const& pipeline, vistk::config_t const& config)
{
  vistk::python::python_gil const gil;

  (void)gil;

  return extract<vistk::schedule_t>(m_obj(pipeline, config));
}
