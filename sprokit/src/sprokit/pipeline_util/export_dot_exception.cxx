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

#include "export_dot_exception.h"

#include <sstream>

/**
 * \file export_dot_exception.cxx
 *
 * \brief Implementation of exceptions used when exporting a pipeline to dot.
 */

namespace sprokit
{

export_dot_exception
::export_dot_exception() noexcept
  : pipeline_exception()
{
}

export_dot_exception
::~export_dot_exception() noexcept
{
}

null_pipeline_export_dot_exception
::null_pipeline_export_dot_exception() noexcept
  : export_dot_exception()
{
  std::stringstream sstr;

  sstr << "The pipeline given to export was NULL";

  m_what = sstr.str();
}

null_pipeline_export_dot_exception
::~null_pipeline_export_dot_exception() noexcept
{
}

null_cluster_export_dot_exception
::null_cluster_export_dot_exception() noexcept
  : export_dot_exception()
{
  std::stringstream sstr;

  sstr << "The cluster given to export was NULL";

  m_what = sstr.str();
}

null_cluster_export_dot_exception
::~null_cluster_export_dot_exception() noexcept
{
}

empty_name_export_dot_exception
::empty_name_export_dot_exception() noexcept
  : export_dot_exception()
{
  std::stringstream sstr;

  sstr << "A process with an empty name cannot be "
          "exported to dot";

  m_what = sstr.str();
}

empty_name_export_dot_exception
::~empty_name_export_dot_exception() noexcept
{
}

}
