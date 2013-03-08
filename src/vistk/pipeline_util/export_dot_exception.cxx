/*ckwg +5
 * Copyright 2011, 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "export_dot_exception.h"

#include <sstream>

/**
 * \file export_dot_exception.cxx
 *
 * \brief Implementation of exceptions used when exporting a pipeline to dot.
 */

namespace vistk
{

export_dot_exception
::export_dot_exception() throw()
  : pipeline_exception()
{
}

export_dot_exception
::~export_dot_exception() throw()
{
}

null_pipeline_export_dot_exception
::null_pipeline_export_dot_exception() throw()
  : export_dot_exception()
{
  std::stringstream sstr;

  sstr << "The pipeline given to export was NULL";

  m_what = sstr.str();
}

null_pipeline_export_dot_exception
::~null_pipeline_export_dot_exception() throw()
{
}

null_cluster_export_dot_exception
::null_cluster_export_dot_exception() throw()
  : export_dot_exception()
{
  std::stringstream sstr;

  sstr << "The cluster given to export was NULL";

  m_what = sstr.str();
}

null_cluster_export_dot_exception
::~null_cluster_export_dot_exception() throw()
{
}

empty_name_export_dot_exception
::empty_name_export_dot_exception() throw()
  : export_dot_exception()
{
  std::stringstream sstr;

  sstr << "A process with an empty name cannot be "
          "exported to dot";

  m_what = sstr.str();
}

empty_name_export_dot_exception
::~empty_name_export_dot_exception() throw()
{
}

}
