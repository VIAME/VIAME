/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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

#include "kwiver_logger_manager.h"

#include "kwiver_logger_factory.h"
#include "default_logger.h"
#include <kwiversys/DynamicLoader.hxx>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>


/*
 * Note: This must be thread safe.
 *
 * Also: In order to make this work, it must be possible to create
 * loggers before the manager has been initialized. This means that
 * the initialization is flexible, adaptive and has a reasonable
 * default.
 */

typedef kwiversys::DynamicLoader DL;

namespace kwiver {
namespace vital {

namespace logger_ns {

  class kwiver_logger_factory;

}


//
// Pointer to our single instance.
//
kwiver_logger_manager* kwiver_logger_manager::s_instance = 0;

#define PLUGIN_ENV_VAR "VITAL_LOGGER_FACTORY"

// ------------------------------------------------------------------
/*
 * Private implememtation
 *
 */
class kwiver_logger_manager::impl
{
public:
  impl() = default;
  ~impl() = default;

  std::unique_ptr< logger_ns::kwiver_logger_factory > m_logFactory;

  // Current library handle
  kwiversys::DynamicLoader::LibraryHandle m_libHandle;

};


// ----------------------------------------------------------------
/** Constructor.
 *
 *
 */
kwiver_logger_manager
::kwiver_logger_manager()
  : m_impl( new impl )
{
  // Need to create a factory class at this point because loggers
  // are created by static initializers. we can wait no longer until
  // we provide a method for creating these loggers.

  bool try_default(false);
  std::string factory_name;
  char const* factory = std::getenv( PLUGIN_ENV_VAR );
  if ( factory == 0 )
  {
    // If no special factory is specified, try default name
    try_default = true;
    factory_name = "vital_logger_plugin";
  }
  else
  {
    factory_name = factory;
    if (*factory_name.rbegin() == ';')
    {
      factory_name = factory_name.substr(0, factory_name.size() - 1);
    }
  }
  factory_name += DL::LibExtension();

  try
  {
    // Dynamically load logger factory.
    load_factory( factory_name );
    return;
  }
  catch( std::runtime_error &e )
  {
    // Only give error if the environment specified logger could not be found
    if ( ! try_default )
    {
      std::cerr << "WARNING: Could not load logger factory \"" << factory_name
                << "\" as specified in environment variable \""
                << PLUGIN_ENV_VAR "\"\n"
                << "Defaulting to built-in logger.\n"
                << e.what() << std::endl;
    }
    else
    {
      std::cerr << "INFO: Could not load default logger factory. Using built-in logger." << std::endl;
    }
  }

  // Create a default logger back end
  m_impl->m_logFactory.reset( new logger_ns::logger_factory_default() );
}


kwiver_logger_manager
::~kwiver_logger_manager()
{ }


// ----------------------------------------------------------------
/** Get singleton instance.
 *
 *
 */
kwiver_logger_manager * kwiver_logger_manager
::instance()
{
  static std::mutex local_lock;          // synchronization lock

  if (0 != s_instance)
  {
    return s_instance;
  }

  std::lock_guard<std::mutex> lock(local_lock);
  if (0 == s_instance)
  {
    // create new object
    s_instance = new kwiver_logger_manager();
  }

  return s_instance;
}


// ----------------------------------------------------------------
/* Get address of logger object.
 *
 * These are unbound functions
 */
VITAL_LOGGER_EXPORT
logger_handle_t
get_logger( char const* name )
{
  return kwiver_logger_manager::instance()->m_impl->m_logFactory->get_logger(name);
}


// ------------------------------------------------------------------
VITAL_LOGGER_EXPORT
logger_handle_t
get_logger( std::string const& name )
{
  return get_logger( name.c_str() );
}


// ------------------------------------------------------------------
VITAL_LOGGER_EXPORT
void
kwiver_logger_manager
::set_logger_factory( std::unique_ptr< logger_ns::kwiver_logger_factory >&& fact )
{
  m_impl->m_logFactory.swap( fact );
}


// ------------------------------------------------------------------
std::string const&
kwiver_logger_manager
::get_factory_name() const
{
  return m_impl->m_logFactory->get_factory_name();
}


// ------------------------------------------------------------------
void
kwiver_logger_manager
::load_factory( std::string const& lib_name )
{
  typedef logger_ns::kwiver_logger_factory* (*FactoryPointer_t)();

  m_impl->m_libHandle = DL::OpenLibrary( lib_name.c_str() );
  if ( ! m_impl->m_libHandle )
  {
    std::stringstream str;
    str << "Unable to load logger factory plug-in: " << DL::LastError();
    throw std::runtime_error( str.str() );
  }

  // Get our entry symbol
  FactoryPointer_t fp = reinterpret_cast< FactoryPointer_t >(
    DL::GetSymbolAddress( m_impl->m_libHandle, "kwiver_logger_factory" ) );
  if ( ! fp )
  {
    DL::CloseLibrary( m_impl->m_libHandle );

    std::stringstream str;
    str << "Unable to bind to function: kwiver_logger_factory() "
        << DL::LastError();
    throw std::runtime_error( str.str() );
  }
  // Get pointer to new logger factory object
  m_impl->m_logFactory.reset( fp() );
}

} } // end namespace
