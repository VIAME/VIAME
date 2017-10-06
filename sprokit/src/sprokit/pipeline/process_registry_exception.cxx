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

#include "process_registry_exception.h"

#include <sstream>

/**
 * \file process_registry_exception.cxx
 *
 * \brief Implementation of exceptions used within the \link sprokit::process_registry process registry\endlink.
 */

namespace sprokit
{

process_registry_exception
::process_registry_exception() noexcept
  : pipeline_exception()
{
}

process_registry_exception
::~process_registry_exception() noexcept
{
}

null_process_ctor_exception
::null_process_ctor_exception(process::type_t const& type) noexcept
  : process_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "A NULL constructor was passed for the "
          "process type \'" << m_type << "\'";

  m_what = sstr.str();
}

null_process_ctor_exception
::~null_process_ctor_exception() noexcept
{
}

null_process_registry_config_exception
::null_process_registry_config_exception() noexcept
  : process_registry_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a process";

  m_what = sstr.str();
}

null_process_registry_config_exception
::~null_process_registry_config_exception() noexcept
{
}

no_such_process_type_exception
::no_such_process_type_exception(process::type_t const& type) noexcept
  : process_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is no such process of type \'" << type << "\' "
          "in the registry";

  m_what = sstr.str();
}

no_such_process_type_exception
::~no_such_process_type_exception() noexcept
{
}

process_type_already_exists_exception
::process_type_already_exists_exception(process::type_t const& type) noexcept
  : process_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is already a process of type \'" << type << "\' "
          "in the registry";

  m_what = sstr.str();
}

process_type_already_exists_exception
::~process_type_already_exists_exception() noexcept
{
}

}
