/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/schedule.h>
#include <vistk/pipeline/schedule_registry.h>
#include <vistk/pipeline/schedule_registry_exception.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

/**
 * \file schedule_registry.cxx
 *
 * \brief Python bindings for \link vistk::schedule_registry\endlink.
 */

using namespace boost::python;

static void register_schedule(vistk::schedule_registry_t self,
                              vistk::schedule_registry::type_t const& type,
                              vistk::schedule_registry::description_t const& desc,
                              object obj);

static void translator(vistk::schedule_registry_exception const& e);

BOOST_PYTHON_MODULE(schedule_registry)
{
  register_exception_translator<
    vistk::schedule_registry_exception>(translator);

  class_<vistk::schedule_registry::type_t>("ScheduleType");
  class_<vistk::schedule_registry::description_t>("ScheduleDescription");
  class_<vistk::schedule_registry::types_t>("ScheduleTypes")
    .def(vector_indexing_suite<vistk::schedule_registry::types_t>())
  ;

  class_<vistk::schedule_registry, vistk::schedule_registry_t, boost::noncopyable>("ScheduleRegistry", no_init)
    .def("self", &vistk::schedule_registry::self)
    .staticmethod("self")
    .def("register_schedule", &register_schedule)
    .def("create_schedule", &vistk::schedule_registry::create_schedule)
    .def("types", &vistk::schedule_registry::types)
    .def("description", &vistk::schedule_registry::description)
  ;
}

class python_schedule_wrapper
{
  public:
    python_schedule_wrapper(object obj);
    ~python_schedule_wrapper();

    vistk::schedule_t operator () (vistk::config_t const& config, vistk::pipeline_t const& pipeline);
  private:
    object const m_obj;
};

void
register_schedule(vistk::schedule_registry_t self,
                  vistk::schedule_registry::type_t const& type,
                  vistk::schedule_registry::description_t const& desc,
                  object obj)
{
  python_schedule_wrapper wrap(obj);

  self->register_schedule(type, desc, wrap);
}

void
translator(vistk::schedule_registry_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
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
::operator () (vistk::config_t const& config, vistk::pipeline_t const& pipeline)
{
  return extract<vistk::schedule_t>(m_obj(config, pipeline));
}
