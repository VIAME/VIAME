/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
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
::edge_exception() noexcept
  : pipeline_exception()
{
}

edge_exception
::~edge_exception() noexcept
{
}

// ----------------------------------------------------------------------------
null_edge_config_exception
::null_edge_config_exception() noexcept
  : edge_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a edge";

  m_what = sstr.str();
}

null_edge_config_exception
::~null_edge_config_exception() noexcept
{
}

// ----------------------------------------------------------------------------
datum_requested_after_complete
::datum_requested_after_complete() noexcept
  : edge_exception()
{
  std::ostringstream sstr;

  sstr << "A datum was requested after "
          "downstream indicated completion";

  m_what = sstr.str();
}

datum_requested_after_complete
::~datum_requested_after_complete() noexcept
{
}

// ----------------------------------------------------------------------------
edge_connection_exception
::edge_connection_exception() noexcept
  : edge_exception()
{
}

edge_connection_exception
::~edge_connection_exception() noexcept
{
}

// ----------------------------------------------------------------------------
null_process_connection_exception
::null_process_connection_exception() noexcept
  : edge_connection_exception()
{
  std::ostringstream sstr;

  sstr << "An edge was given a NULL pointer to connect";

  m_what = sstr.str();
}

null_process_connection_exception
::~null_process_connection_exception() noexcept
{
}

// ----------------------------------------------------------------------------
duplicate_edge_connection_exception
::duplicate_edge_connection_exception(process::name_t const& name, process::name_t const& new_name, std::string const& type) noexcept
  : edge_connection_exception()
  , m_name(name)
  , m_new_name(new_name)
{
  std::ostringstream sstr;

  sstr << "An edge was given a process for the "
       << type << " input (\'" << m_new_name
       << "\') when one already exists (\'" << m_name << "\')";

  m_what = sstr.str();
}

duplicate_edge_connection_exception
::~duplicate_edge_connection_exception() noexcept
{
}

// ----------------------------------------------------------------------------
input_already_connected_exception
::input_already_connected_exception(process::name_t const& name, process::name_t const& new_name) noexcept
  : duplicate_edge_connection_exception(name, new_name, "input")
{
}

input_already_connected_exception
::~input_already_connected_exception() noexcept
{
}

// ----------------------------------------------------------------------------
output_already_connected_exception
::output_already_connected_exception(process::name_t const& name, process::name_t const& new_name) noexcept
  : duplicate_edge_connection_exception(name, new_name, "output")
{
}

output_already_connected_exception
::~output_already_connected_exception() noexcept
{
}

}
