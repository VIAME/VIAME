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

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_factory.h>
#include <sprokit/pipeline/scheduler_registry_exception.h>

#include <sprokit/python/util/python_threading.h>
#include <sprokit/python/util/python_gil.h>
#include <sprokit/python/util/python_exceptions.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <pybind11/stl_bind.h>

#ifdef WIN32
 // Windows get_pointer const volatile workaround
namespace boost
{
  template <> inline sprokit::scheduler const volatile*
  get_pointer(class sprokit::scheduler const volatile* p)
  {
    return p;
  }
  template <> inline sprokit::scheduler_factory const volatile*
  get_pointer(class sprokit::scheduler_factory const volatile* p)
  {
    return p;
  }
}
#endif

using namespace pybind11;

static void register_scheduler( sprokit::scheduler::type_t const& type,
                                sprokit::scheduler::description_t const& desc,
                                object obj );
static bool is_scheduler_loaded( const std::string& name );
static void mark_scheduler_loaded( const std::string& name );
static std::string get_description( const std::string& name );
static std::vector< std::string > scheduler_names();
static std::string get_default_type();

//==================================================================
PYBIND11_MODULE(scheduler_factory, m)
{

  bind_vector<std::vector< std::string > >(m, "string_vector");

  // Define unbound functions.
  m.def("add_scheduler", &register_scheduler
      , arg("type"), arg("description"), arg("ctor")
      , "Registers a function which creates a scheduler of the given type."
      , return_value_policy::reference_internal);

  m.def("create_scheduler", &sprokit::create_scheduler
      , arg("type"), arg("pipeline"), arg("config") = kwiver::vital::config_block::empty_config()
      , "Creates a new scheduler of the given type.");

  m.def("is_scheduler_module_loaded", &is_scheduler_loaded
      , (arg("module"))
      , "Returns True if the module has already been loaded, False otherwise.");

  m.def("mark_scheduler_module_as_loaded", &mark_scheduler_loaded
      , (arg("module"))
      , "Marks a module as loaded.");

  m.def("types", &scheduler_names
      , "A list of known scheduler types.");

  m.def("description", &get_description
      , (arg("type"))
      , "The description for the given scheduler type.");

  m.def("default_type", &get_default_type
      , "The default scheduler type.");

  m.attr("Scheduler") = m.import("sprokit.pipeline.scheduler").attr("PythonScheduler");

}


class python_scheduler_wrapper
  : sprokit::python::python_threading
{
  public:
    python_scheduler_wrapper(object obj);
    ~python_scheduler_wrapper();

    object operator () (sprokit::pipeline_t const& pipeline,
                                      kwiver::vital::config_block_sptr const& config);
  private:
    object const m_obj;
};



void
register_scheduler( sprokit::scheduler::type_t const& type,
                    sprokit::scheduler::description_t const& desc,
                    object obj )
{
  python_scheduler_wrapper const wrap(obj);

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  auto fact = vpm.add_factory( new sprokit::python_scheduler_factory( type,
                                                             typeid( object ).name(),
                                                             wrap ) );

  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, type )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, "python-runtime" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, desc );
}


// ------------------------------------------------------------------
bool is_scheduler_loaded( const std::string& name )
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  return vpm.is_module_loaded( name );
}


// ------------------------------------------------------------------
void mark_scheduler_loaded( const std::string& name )
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.mark_module_as_loaded( name );
}


// ------------------------------------------------------------------
std::string get_description( const std::string& type )
{
  kwiver::vital::plugin_factory_handle_t a_fact;

  try
  {
    typedef kwiver::vital::implementation_factory_by_name< sprokit::scheduler > proc_factory;
    proc_factory ifact;

    SPROKIT_PYTHON_TRANSLATE_EXCEPTION(
      a_fact = ifact.find_factory( type );
      )
  }
  catch ( const std::exception &e )
  {
    typedef kwiver::vital::implementation_factory_by_name< object > proc_factory;
    proc_factory ifact;

    SPROKIT_PYTHON_TRANSLATE_EXCEPTION(
      a_fact = ifact.find_factory( type );
      )
  }


  std::string buf = "-- Not Set --";
  a_fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, buf );

  return buf;
}


// ------------------------------------------------------------------
std::vector< std::string > scheduler_names()
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  auto py_fact_list = vpm.get_factories<object>();

  std::vector<std::string> name_list;
  for( auto fact : py_fact_list )
  {
    std::string buf;
    if (fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, buf ))
    {
      name_list.push_back( buf );
    }
  } // end foreach

  auto fact_list = vpm.get_factories<sprokit::scheduler>();
  for( auto fact : fact_list )
  {
    std::string buf;
    if (fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, buf ))
    {
      name_list.push_back( buf );
    }
  } // end foreach

  return name_list;
}


// ------------------------------------------------------------------
std::string get_default_type()
{
  return sprokit::scheduler_factory::default_type;
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
object
python_scheduler_wrapper
::operator () (sprokit::pipeline_t const& pipeline, kwiver::vital::config_block_sptr const& config)
{
  return m_obj(pipeline, config);
}
