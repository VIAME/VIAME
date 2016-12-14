/*ckwg +29
 * Copyright 2011-2012 by Kitware, Inc.
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

/**
 * \file scheduler_factory.cxx
 *
 * \brief Python bindings for \link sprokit::scheduler_factory\endlink.
 */

#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_factory.h>

#include <sprokit/python/util/python_threading.h>
#include <sprokit/python/util/python_gil.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/wrapper.hpp>
#include <boost/python/def.hpp>

using namespace boost::python;

static void register_scheduler( sprokit::scheduler::type_t const& type,
                                sprokit::scheduler::description_t const& desc,
                                object obj );

BOOST_PYTHON_MODULE(scheduler_factory)
{
  def("add_scheduler", &register_scheduler
      , (arg("type"), arg("description"), arg("ctor"))
      , "Registers a function which creates a scheduler of the given type.");

  def("create_scheduler", &sprokit::create_scheduler
      , (arg("type"), arg("pipeline"), arg("config") = kwiver::vital::config_block::empty_config())
      , "Creates a new scheduler of the given type.");


  def("is_scheduler module_loaded", &sprokit::is_scheduler_module_loaded
      , (arg("module"))
      , "Returns True if the module has already been loaded, False otherwise.");

  def("mark_scheduler_module_as_loaded", &sprokit::mark_scheduler_module_as_loaded
      , (arg("module"))
      , "Marks a module as loaded.");


  class_<sprokit::scheduler::type_t>("SchedulerType"
    , "The type for a type of scheduler.");

  class_<kwiver::vital::plugin_manager::module_t>("SchedulerModule"
    , "The type for a scheduler module name.");

  class_<sprokit::scheduler, sprokit::scheduler_t, boost::noncopyable>("Scheduler"
    , "An abstract class which offers an interface for pipeline execution strategies."
    , no_init)
    .def("start", &sprokit::scheduler::start
      , "Start the execution of the pipeline.")
    .def("wait", &sprokit::scheduler::wait
      , "Wait until the pipeline execution is complete.")
    .def("stop", &sprokit::scheduler::stop
      , "Stop the execution of the pipeline.")
  ;

  class_<sprokit::scheduler_factory, sprokit::scheduler_factory, boost::noncopyable>("SchedulerFactory"
    , "A registry of all known scheduler types."
    , no_init)

    /*
    .def("types", &sprokit::scheduler_registry::types //+ fixme - part of plugin manager
      , "A list of known scheduler types.")

    .def("description", &sprokit::scheduler_registry::description //+ part of plugin manager
      , (arg("type"))
      , "The description for the given scheduler type.")
    */
    .def_readonly("default_type", &sprokit::scheduler_factory::default_type
      , "The default scheduler type.")
  ;
}


class python_scheduler_wrapper
  : sprokit::python::python_threading
{
  public:
    python_scheduler_wrapper(object obj);
    ~python_scheduler_wrapper();

    sprokit::scheduler_t operator () (sprokit::pipeline_t const& pipeline,
                                      kwiver::vital::config_block_sptr const& config);
  private:
    object const m_obj;
};


// ------------------------------------------------------------------
void
register_scheduler( sprokit::scheduler::type_t const& type,
                    sprokit::scheduler::description_t const& desc,
                    object obj )
{
  sprokit::python::python_gil const gil;

  (void)gil;

  python_scheduler_wrapper const wrap(obj);

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  sprokit::scheduler::type_t derived_type = "python::";
  auto fact = vpm.add_factory( new sprokit::scheduler_factory( derived_type + type, // derived type name string
                                                               type, // name of the scheduler
                                                               wrap ) );

  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, type )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, "python-runtime" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, desc );
}


// ------------------------------------------------------------------
python_scheduler_wrapper
::python_scheduler_wrapper(object obj)
  : m_obj(obj)
{
}

python_scheduler_wrapper
::~python_scheduler_wrapper()
{
}


// ------------------------------------------------------------------
sprokit::scheduler_t
python_scheduler_wrapper
::operator () (sprokit::pipeline_t const& pipeline, kwiver::vital::config_block_sptr const& config)
{
  sprokit::python::python_gil const gil;

  (void)gil;

  return extract<sprokit::scheduler_t>(m_obj(pipeline, config));
}
