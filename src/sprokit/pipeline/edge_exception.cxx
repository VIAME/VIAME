/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "edge_exception.h"

#include <sstream>

/**
 * \file edge_exception.cxx
 *
 * \brief Implementation of exceptions used within \link sprokit::edge edges\endlink.
 */

namespace sprokit
{

edge_exception
::edge_exception() SPROKIT_NOTHROW
  : pipeline_exception()
{
}

edge_exception
::~edge_exception() SPROKIT_NOTHROW
{
}

null_edge_config_exception
::null_edge_config_exception() SPROKIT_NOTHROW
  : edge_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a edge";

  m_what = sstr.str();
}

null_edge_config_exception
::~null_edge_config_exception() SPROKIT_NOTHROW
{
}

datum_requested_after_complete
::datum_requested_after_complete() SPROKIT_NOTHROW
  : edge_exception()
{
  std::ostringstream sstr;

  sstr << "A datum was requested after "
          "downstream indicated completion";

  m_what = sstr.str();
}

datum_requested_after_complete
::~datum_requested_after_complete() SPROKIT_NOTHROW
{
}

edge_connection_exception
::edge_connection_exception() SPROKIT_NOTHROW
  : edge_exception()
{
}

edge_connection_exception
::~edge_connection_exception() SPROKIT_NOTHROW
{
}

null_process_connection_exception
::null_process_connection_exception() SPROKIT_NOTHROW
  : edge_connection_exception()
{
  std::ostringstream sstr;

  sstr << "An edge was given a NULL pointer to connect";

  m_what = sstr.str();
}

null_process_connection_exception
::~null_process_connection_exception() SPROKIT_NOTHROW
{
}

duplicate_edge_connection_exception
::duplicate_edge_connection_exception(process::name_t const& name, process::name_t const& new_name, std::string const& type) SPROKIT_NOTHROW
  : edge_connection_exception()
  , m_name(name)
  , m_new_name(new_name)
{
  std::ostringstream sstr;

  sstr << "An edge was given a process for the "
       << type << " input "
          "(\'" << m_new_name << "\') when one already "
          "exists (\'" << m_name << "\')";

  m_what = sstr.str();
}

duplicate_edge_connection_exception
::~duplicate_edge_connection_exception() SPROKIT_NOTHROW
{
}

input_already_connected_exception
::input_already_connected_exception(process::name_t const& name, process::name_t const& new_name) SPROKIT_NOTHROW
  : duplicate_edge_connection_exception(name, new_name, "input")
{
}

input_already_connected_exception
::~input_already_connected_exception() SPROKIT_NOTHROW
{
}

output_already_connected_exception
::output_already_connected_exception(process::name_t const& name, process::name_t const& new_name) SPROKIT_NOTHROW
  : duplicate_edge_connection_exception(name, new_name, "output")
{
}

output_already_connected_exception
::~output_already_connected_exception() SPROKIT_NOTHROW
{
}

}
