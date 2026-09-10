// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief Factory registration function for exposing python-defined
 * implementations.
 */

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

// #include <python/kwiver/internal/python.h>
#include <python/kwiver/internal/python_plugin_factory.h>

#include <kwiversys/Encoding.hxx>
#include <kwiversys/SystemTools.hxx>
#include <python/kwiver/vital/plugins/plugins_from_python_export.h>

#if defined( _WIN32 ) && !defined( __CYGWIN__ )
#include <windows.h>
#ifndef _MAX_PATH
#define _MAX_PATH 4096
#endif
#endif

#include <cstdlib>
#include <exception>
#include <filesystem>

namespace kv = ::kwiver::vital;
namespace py = pybind11;

// ----------------------------------------------------------------------------
// Helper Functions

/**
 * @brief Initialize a python interpreter if one is not already.
 *
 * This function is idempotent.
 */
static bool check_and_initialize_python_interpreter();

/**
 * @brief Body of the registration function, called with exceptions contained.
 */
static void register_factories_impl( ::kv::plugin_loader& vpl );

// ----------------------------------------------------------------------------
// Registration Function
extern "C"
[[maybe_unused]] PLUGINS_FROM_PYTHON_EXPORT

void
register_factories( ::kv::plugin_loader& vpl )
{
  ::kv::logger_handle_t log = ::kv::get_logger(
    "python.kwiver.vital.plugins.register_factories"
  );

  // The plugin loader calls this through a function pointer, so an exception
  // leaving here unwinds out of an extern "C" boundary and reaches
  // std::terminate -- killing every process that loads this plugin, kwiver's
  // own tools included. Python plugins are an optional extra: if discovery
  // fails, say the interpreter cannot import the discovery module, the host
  // should carry on without them.
  try
  {
    register_factories_impl( vpl );
  }
  catch( std::exception const& e )
  {
    LOG_ERROR(
      log, "Python plugin discovery failed, continuing without python "
      "plugins: " << e.what() );
  }
  catch( ... )
  {
    LOG_ERROR(
      log, "Python plugin discovery failed with an unrecognized exception, "
      "continuing without python plugins" );
  }
}

void
register_factories_impl( ::kv::plugin_loader& vpl )
{
  ::kv::logger_handle_t log = ::kv::get_logger(
    "python.kwiver.vital.plugins.register_factories"
  );

  // TODO: Hook to skip python plugin registration.
  //   !!! This would explicitly be before potentially starting the
  //       interpreter. Base on the presence of an environment variable.
  //       Was previously "SPROKIT_NO_PYTHON_MODULES", should be something more
  //       applicable (this isn't sprokit specific).

  // Make sure there is an interpreter running.  If none can be initialized
  // (for example a C++ host with no Python environment configured), skip
  // Python plugin discovery rather than importing on a dead interpreter.
  if( !check_and_initialize_python_interpreter() )
  {
    LOG_WARN(
      log,
      "No Python interpreter available; skipping Python plugin discovery" );
    return;
  }

  // In upstream, in this slot there was logic to dynamically load the
  // `libpython*.so` library here.
  // Ameya explained anecdotally that without this, the following plugin
  // loading block will raise segfaults due to symbol not found errors.
  // If we find this to still be true, then reinstate this logic.
  // * First, just reinstate the portion that loaded the library as introspected
  //   from the interpreter, NOT from the environment PYTHON_LIBRARY variable.
  // * Upstream code specifically used unix `dlopen` instead of KWSYS tool in
  //   order to pass the `RTLD_GLOBAL` flag that KWSYS does not. Ameya reported
  //   getting faults otherwise.
  //   * Modify local version of KWSYS to also pass the `RTLD_GLOBAL` flag?

  // Generate factories to add to `vpl`.
  // Acquire the GIL since we need to call Python functions.
  // The GIL may have been released by a previous Python initialization.
  py::gil_scoped_acquire gil;

  py::object const mod_discovery =
    py::module::import( "kwiver.vital.plugins.discovery" );

  py::list python_concrete_vec =
    mod_discovery.attr( "_get_concrete_pluggable_types" )();

  for( size_t i = 0; i < python_concrete_vec.size(); ++i )
  {
    py::object o = python_concrete_vec[ i ];
    LOG_DEBUG( log,
      "Registering factory for python impl for interface \"" <<
        o.attr("interface_name")().cast< std::string > () << "\": \"" <<
        o.attr("__name__").cast< std::string > () << "\"");

      auto* fact = new kv::python::python_plugin_factory( o );
      vpl.add_factory( fact );
  }
}

/**
 * Returns the name of the library providing the symbol. For example, if you
 * want to locate where the python libraries located call
 * `GetLibraryPathForSymbolWin32(PyInitialize)` on Windows.
 * Taken from vtkResourceFileLocator
 */
#if defined( _WIN32 ) && !defined( __CYGWIN__ )

std::string
GetLibraryPathForSymbolWin32( const void* fptr )
{
  MEMORY_BASIC_INFORMATION mbi;
  VirtualQuery( fptr, &mbi, sizeof( mbi ) );

  wchar_t pathBuf[ _MAX_PATH ];
  if( !GetModuleFileNameW(
    static_cast< HMODULE >( mbi.AllocationBase ),
    pathBuf, sizeof( pathBuf ) ) )
  {
    return std::string();
  }

  return kwiversys::Encoding::ToNarrow( pathBuf );
}

#endif

std::string
get_virtual_env_site_packages()
{
  std::string site_packages_path;
  // Based on Python's documentation the enviromental VIRTUAL_ENV is set to
  // the virtual environment's path when it is activated.  path naming is
  // also standarized even though PY_MAJOR_VERSION and PY_MINOR_VERSION are
  // compile time they should match because  we use the same version to
  // compile and run.  Note: If at some point we use stable abi (abi3) wheels
  // we would need to adapt the code extract the version on runtime instead
  // of compile time.
  if( const char* virtual_env = std::getenv( "VIRTUAL_ENV" ) )
  {
#if defined( _WIN32 )
    site_packages_path = std::string( virtual_env ) + "\\Lib\\site-packages";
#else
    site_packages_path = std::string( virtual_env ) + "/lib/python" +
                         std::to_string( PY_MAJOR_VERSION ) + "." +
                         std::to_string( PY_MINOR_VERSION ) + "/site-packages";
#endif
  }
  return site_packages_path;
}

// Calculate PYTHONHOME based on the library we are linked to.
// Python expects to find system packages under PYTHONHOME\\Lib on Windows.
// It looks like calculating this path is required for windows.
std::string
get_python_home()
{
#if defined( _WIN32 ) && !defined( __CYGWIN__ )
  // Find the library we linked to get the symbol
  auto home_path =
    std::filesystem::path(
      GetLibraryPathForSymbolWin32(
        reinterpret_cast< const void* >( &Py_Initialize ) ) ).parent_path();
  return home_path.string();
#endif
  return "";
}

// the argument of config.home needs to be in static storage based on Python
// documentation.
static std::wstring pythonHome;

// ----------------------------------------------------------------------------
// Helper function implementations
bool
check_and_initialize_python_interpreter()
{
  ::kv::logger_handle_t log = ::kv::get_logger(
    "python.kwiver.vital.plugins.check_and_initialize_python_interpreter"
  );

  // Tracks whether this process embedded the interpreter itself, as opposed to
  // kwiver having been loaded from a running python. Only in the former case
  // is the GIL ours to hand back further down.
  static bool interpreter_is_ours = false;

  // Check if a python interpreter already exists, so we don't clobber sys.argv
  // (e.g. if sprokit is initialized from python)
  if( !Py_IsInitialized() )
  {
    // Embed a python interpreter if one does not exist
    LOG_DEBUG( log, "Initializing python interpreter" );

    const std::string home_path = get_python_home();
    pythonHome = kwiversys::Encoding::ToWide( home_path );

    PyConfig config;
    PyStatus status;
    PyConfig_InitPythonConfig( &config );
    if( !pythonHome.empty() )
    {
      config.home = pythonHome.data();
    }
    // Set Python interpreter attribute: sys.argv = []
    // parameters are: (argc, argv)
    status = PyConfig_SetArgv( &config, 0, nullptr );
    if( PyStatus_IsError( status ) )
    {
      PyConfig_Clear( &config );
      LOG_ERROR( log, "Error setting the configuration of Py_Initialize" );
      return false;
    }

    status = Py_InitializeFromConfig( &config );
    if( PyStatus_IsError( status ) )
    {
      PyConfig_Clear( &config );
      LOG_ERROR( log, "Error calling Py_Initialize" );
      return false;
    }
    PyConfig_Clear( &config );
    LOG_DEBUG( log, "Python interpreter initialized" );
    interpreter_is_ours = true;
  }

  // if we are in a virtual enviroment add its site-packages in the module
  // search paths of the interpreter.
  //
  // Only the first call through here does so. This function runs again for
  // every plugin load, and by then the GIL has been handed back by
  // PyEval_SaveThread() below, so the GIL has to be reacquired before touching
  // the interpreter -- otherwise the second load faults.
  static bool virtual_env_path_added = false;
  std::string const virtual_env_site_packages_path =
    get_virtual_env_site_packages();
  if( !virtual_env_path_added && !virtual_env_site_packages_path.empty() )
  {
    virtual_env_path_added = true;

    LOG_DEBUG(
      log,
      "Adding " << virtual_env_site_packages_path << " to pythonpath" );

    py::gil_scoped_acquire gil;

    PyObject* sys_path = PySys_GetObject( "path" );
    PyObject* item =
      PyUnicode_FromStringAndSize(
        virtual_env_site_packages_path.c_str(),
        virtual_env_site_packages_path.size() );
    if( !sys_path || !item )
    {
      LOG_ERROR( log, "Error getting sys.path" );
      PyErr_Clear();
    }
    else if( PyList_Insert( sys_path, 0, item ) )
    {
      LOG_ERROR( log, "Error prepending item to sys.path" );
      PyErr_Clear();
    }
    Py_XDECREF( item );
  }

  // Let pybind11 initialize threads and set up its internal data structures if
  // not already done so.
  //
  // This is deliberately not guarded on PyEval_ThreadsInitialized(): since
  // Python 3.7 threads are always initialized by the time the interpreter is
  // up, so that function always returns true and the release below would never
  // run. The GIL would then stay held by whichever thread got here first --
  // the main thread, for a tool like the pipeline runner -- and every
  // subsequent attempt to call into Python from another thread would block in
  // gil_scoped_acquire forever. That is only visible with a scheduler that
  // steps processes on its own threads, so a pipeline with a python algorithm
  // deadlocks under thread_per_process while working under
  // pythread_per_process.
  //
  // Guard on having done it once instead, and only hand back the GIL if this
  // thread actually holds it: when kwiver is loaded from a python process the
  // caller owns the GIL and expects it back.
  static bool threads_initialized = false;

  if( interpreter_is_ours && !threads_initialized )
  {
    threads_initialized = true;

    LOG_DEBUG(
      log, "Letting pybind11 initialize its internal data structures." );
    {
      pybind11::detail::get_internals();
    }

    // Release the GIL so other threads can acquire it
    PyEval_SaveThread();
  }
  return true;
}
