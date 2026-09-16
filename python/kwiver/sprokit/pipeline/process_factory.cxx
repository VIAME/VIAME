// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file process_factory.cxx
 *
 * \brief Python bindings for \link viame::pipeline::process_factory\endlink.
 */

#include <viame/pipeline_framework/process.h>
#include <viame/pipeline_framework/process_factory.h>
#include <viame/pipeline_framework/process_registry_exception.h>

#include <python/kwiver/vital/util/python_exceptions.h>

#include <viame/algorithm_framework/plugin/plugin_manager.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "python_wrappers.cxx"

using namespace pybind11;

// We need our own factory for inheritance to work
// This is hopefully something pybind11 will deal with soon, and we can
// eliminate this class
// Otherwise, we can rewrite process_factory to have multiple entrypoints
namespace viame {

namespace pipeline {

namespace python {

static void register_process(
  ::viame::pipeline::process::type_t const& type,
  ::viame::pipeline::process::description_t const& desc,
  object obj );

static bool is_process_loaded( const std::string& name );
static void mark_process_loaded( const std::string& name );
static std::string get_description( const std::string& name );
static std::vector< std::string > process_names();

// ============================================================================
typedef std::function< pybind11::object ( viame::config_block_sptr const& config ) >
  py_process_factory_func_t;

class python_process_factory
  : public ::viame::pipeline::process_factory
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
  python_process_factory(
    const std::string& type,
    const std::string& itype,
    py_process_factory_func_t factory );

  virtual ~python_process_factory();

  virtual ::viame::pipeline::process_t create_object(
    viame::config_block_sptr const& config );

private:
  py_process_factory_func_t m_factory;
};

// ------------------------------------------------------------------
python_process_factory
::python_process_factory(
  const std::string& type,
  const std::string& itype,
  py_process_factory_func_t factory )
  : process_factory( type, itype ),
    m_factory( factory )
{
  this->add_attribute( CONCRETE_TYPE, type )
    .add_attribute( PLUGIN_FACTORY_TYPE, typeid( *this ).name() )
    .add_attribute( PLUGIN_CATEGORY, "process" )
    .add_attribute( PLUGIN_NAME, type )
    .add_attribute( PLUGIN_MODULE_NAME, "python-runtime" );
}

python_process_factory::
~python_process_factory()
{}

// ----------------------------------------------------------------------------
::viame::pipeline::process_t
python_process_factory
::create_object( viame::config_block_sptr const& config )
{
  pybind11::gil_scoped_acquire acquire;
  ( void ) acquire;

  // Call sprokit factory function.
  pybind11::object obj = m_factory( config );

  // We need to do it this way because of how pybind11 handles memory
  obj.inc_ref();
  ::viame::pipeline::process_t proc_ptr = obj.cast< ::viame::pipeline::process_t >();
  return proc_ptr;
}

} // namespace python

} // namespace pipeline

} // namespace viame

using namespace viame::pipeline::python;

// ==================================================================
PYBIND11_MODULE( process_factory, m )
{
  class_< viame::pipeline::processes_t >(
    m, "Processes",
    "A collection of processes." );

  bind_vector< std::vector< std::string > >( m, "StringVector" );

  m.def(
    "is_process_module_loaded", &is_process_loaded,
    call_guard< pybind11::gil_scoped_release >(),
    ( arg( "module" ) ),
    "Returns True if the module has already been loaded, False otherwise." );

  m.def(
    "mark_process_module_as_loaded", &mark_process_loaded,
    call_guard< pybind11::gil_scoped_release >(),
    ( arg( "module" ) ),
    "Marks a module as loaded." );

  m.def(
    "add_process", &register_process,
    call_guard< pybind11::gil_scoped_release >(),
    arg( "type" ), arg( "description" ), arg( "ctor" ),
    "Registers a function which creates a process of the given type." );

  m.def(
    "create_process", &viame::pipeline::create_process,
    call_guard< pybind11::gil_scoped_release >(),
    arg( "type" ), arg( "name" ),
    arg( "config" ) = viame::config_block::empty_config(),
    "Creates a new process of the given type.",
    return_value_policy::reference_internal );

  m.def(
    "description", &get_description,
    call_guard< pybind11::gil_scoped_release >(),
    ( arg( "type" ) ),
    "Returns description for the process" );

  m.def(
    "types", &process_names, call_guard< pybind11::gil_scoped_release >(),
    "Returns list of process names" );

  m.attr( "Process" ) =
    m.import( "kwiver.sprokit.pipeline.process" ).attr( "PythonProcess" );
}

namespace viame {

namespace pipeline {

namespace python {

// ==================================================================
class python_process_wrapper
{
public:
  python_process_wrapper( object obj );
  ~python_process_wrapper();

  object operator()( viame::config_block_sptr const& config );

private:
  object const m_obj;
};

// ------------------------------------------------------------------
void
register_process(
  ::viame::pipeline::process::type_t const&        type,
  ::viame::pipeline::process::description_t const& desc,
  object obj )
{
  pybind11::gil_scoped_acquire acquire;
  ( void ) acquire;

  python_process_wrapper const& wrap( obj );

  viame::plugin_manager& vpm =
    viame::plugin_manager::instance();
  auto fact = vpm.add_factory(
    new python_process_factory(
      type,
      ::viame::pipeline::process::interface_name(),
      wrap ) );

  fact->add_attribute(
    viame::plugin_factory::PLUGIN_DESCRIPTION,
    desc );
}

// ------------------------------------------------------------------
bool
is_process_loaded( const std::string& name )
{
  viame::plugin_manager& vpm =
    viame::plugin_manager::instance();
  return vpm.is_module_loaded( name );
}

// ------------------------------------------------------------------
void
mark_process_loaded( const std::string& name )
{
  viame::plugin_manager& vpm =
    viame::plugin_manager::instance();
  vpm.mark_module_as_loaded( name );
}

// ------------------------------------------------------------------
std::string
get_description( const std::string& type )
{
  viame::plugin_factory_handle_t a_fact;

  // Python processes are registered with viame::pipeline::process interface type
  // (see register_process above), so we only need to look up using that type.
  typedef viame::implementation_factory_by_name< ::viame::pipeline::process >
    proc_factory;

  proc_factory ifact;

  VITAL_PYTHON_TRANSLATE_EXCEPTION(
    a_fact = ifact.find_factory( type );
  )

  std::string buf = "-- Not Set --";
  a_fact->get_attribute(
    viame::plugin_factory::PLUGIN_DESCRIPTION,
    buf );

  return buf;
}

// ------------------------------------------------------------------
std::vector< std::string >
process_names()
{
  std::vector< std::string > name_list;

  auto fact_list = ::viame::pipeline::get_process_list();
  for( auto fact : fact_list )
  {
    std::string buf;
    if( fact->get_attribute( viame::plugin_factory::PLUGIN_NAME, buf ) )
    {
      name_list.push_back( buf );
    }
  } // end foreach

  return name_list;
}

// ------------------------------------------------------------------
python_process_wrapper
::python_process_wrapper( object obj )
  : m_obj( object( obj ) )
{}

python_process_wrapper
::~python_process_wrapper()
{}

// ------------------------------------------------------------------
object
python_process_wrapper
::operator()( viame::config_block_sptr const& config )
{
  pybind11::gil_scoped_acquire acquire;
  ( void ) acquire;
  return m_obj( config );
}

} // namespace python

} // namespace pipeline

} // namespace viame
