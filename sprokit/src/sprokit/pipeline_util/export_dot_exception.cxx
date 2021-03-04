// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
