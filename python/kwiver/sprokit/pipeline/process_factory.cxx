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

/**
 * \file process_factory.cxx
 *
 * \brief Python bindings for \link sprokit::process_factory\endlink.
 */

#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/process_registry_exception.h>

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/util/python_exceptions.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <pybind11/stl_bind.h>
#include "python_wrappers.cxx"

using namespace pybind11;

// We need our own factory for inheritance to work
// This is hopefully something pybind11 will deal with soon, and we can eliminate this class
// Otherwise, we can rewrite process_factory to have multiple entrypoints

static void register_process( sprokit::process::type_t const& type,
                              sprokit::process::description_t const& desc,
                              object obj );

static bool is_process_loaded( const std::string& name );
static void mark_process_loaded( const std::string& name );
static std::string get_description( const std::string& name );
static std::vector< std::string > process_names();

// ============================================================================
typedef std::function< pybind11::object( kwiver::vital::config_block_sptr const& config ) > py_process_factory_func_t;

class python_process_factory
  : public sprokit::process_factory
{
  /**
   * @brief CTOR for factory object
   *
   * This CTOR is designed to work in conjunction with pybind11
   *
   * @param type Type name of the process
   * @param itype Type name of interface type.
   * @param factory The Factory function
   */
  public:

  python_process_factory( const std::string& type,
                          const std::string& itype,
                          py_process_factory_func_t factory );

  virtual ~python_process_factory();

  virtual sprokit::process_t create_object(kwiver::vital::config_block_sptr const& config);

private:
  py_process_factory_func_t m_factory;
};


// ------------------------------------------------------------------
python_process_factory::
python_process_factory( const std::string& type,
                        const std::string& itype,
                        py_process_factory_func_t factory )
  : process_factory( type, itype )
  , m_factory( factory )
{
  this->add_attribute( CONCRETE_TYPE, type)
    .add_attribute( PLUGIN_FACTORY_TYPE, typeid(* this ).name() )
    .add_attribute( PLUGIN_CATEGORY, "process" );
}

python_process_factory::
~python_process_factory()
{ }


// ----------------------------------------------------------------------------
sprokit::process_t
python_process_factory::
create_object(kwiver::vital::config_block_sptr const& config)
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  // Call sprokit factory function.
  pybind11::object obj = m_factory(config);

  // We need to do it this way because of how pybind11 handles memory
  obj.inc_ref();
  sprokit::process_t proc_ptr = obj.cast<sprokit::process_t>();
  return proc_ptr;
}


// ==================================================================
PYBIND11_MODULE(process_factory, m)
{
  class_<sprokit::processes_t>(m, "Processes"
    , "A collection of processes.");

  bind_vector<std::vector<std::string> >(m, "StringVector");

  m.def("is_process_module_loaded", &is_process_loaded, call_guard<kwiver::vital::python::gil_scoped_release>()
       , (arg("module"))
       , "Returns True if the module has already been loaded, False otherwise.");

  m.def("mark_process_module_as_loaded", &mark_process_loaded, call_guard<kwiver::vital::python::gil_scoped_release>()
       , (arg("module"))
       , "Marks a module as loaded.");

  m.def("add_process", &register_process, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("type"), arg("description"), arg("ctor")
       , "Registers a function which creates a process of the given type.");

  m.def("create_process", &sprokit::create_process, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("type"), arg("name"), arg("config") = kwiver::vital::config_block::empty_config()
      , "Creates a new process of the given type.", return_value_policy::reference_internal);

  m.def("description", &get_description, call_guard<kwiver::vital::python::gil_scoped_release>()
       , (arg("type"))
       , "Returns description for the process");

  m.def("types", &process_names, call_guard<kwiver::vital::python::gil_scoped_release>()
       , "Returns list of process names" );

  m.attr("Process") = m.import("kwiver.sprokit.pipeline.process").attr("PythonProcess");
  m.attr("ProcessCluster") = m.import("kwiver.sprokit.pipeline.process_cluster").attr("PythonProcessCluster");

}

// ==================================================================
class python_process_wrapper
{
public:
  python_process_wrapper( object obj );
  ~python_process_wrapper();

  object operator()( kwiver::vital::config_block_sptr const& config );


private:
  object const m_obj;
};


// ------------------------------------------------------------------
void
register_process( sprokit::process::type_t const&        type,
                  sprokit::process::description_t const& desc,
                  object                                 obj )
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  python_process_wrapper const& wrap(obj);

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  auto fact = vpm.add_factory( new python_process_factory( type, // derived type name string
                                                           typeid( sprokit::process ).name(),
                                                           wrap ) );

  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, type )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, "python-runtime" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, desc )
    ;
}


// ------------------------------------------------------------------
bool is_process_loaded( const std::string& name )
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  return vpm.is_module_loaded( name );
}


// ------------------------------------------------------------------
void mark_process_loaded( const std::string& name )
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
    typedef kwiver::vital::implementation_factory_by_name< sprokit::process > proc_factory;
    proc_factory ifact;

    VITAL_PYTHON_TRANSLATE_EXCEPTION(
      a_fact = ifact.find_factory( type );
      )

  }
  catch ( const std::exception &e )
  {
    typedef kwiver::vital::implementation_factory_by_name< object > py_proc_factory;
    py_proc_factory ifact;

    VITAL_PYTHON_TRANSLATE_EXCEPTION(
      a_fact = ifact.find_factory( type );
      )
  }


  std::string buf = "-- Not Set --";
  a_fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, buf );

  return buf;
}


// ------------------------------------------------------------------
std::vector< std::string > process_names()
{
  std::vector< std::string > name_list;

  auto fact_list = sprokit::get_process_list();
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
python_process_wrapper
  ::python_process_wrapper( object obj )
  : m_obj( object(obj) )
{
}


python_process_wrapper
  ::~python_process_wrapper()
{
}


// ------------------------------------------------------------------
object
python_process_wrapper
  ::operator()( kwiver::vital::config_block_sptr const& config )
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;
  return m_obj( config );
}
