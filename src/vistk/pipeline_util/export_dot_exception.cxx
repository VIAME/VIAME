/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
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

}
