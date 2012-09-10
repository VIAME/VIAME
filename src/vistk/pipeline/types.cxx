/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "types.h"

/**
 * \file types.cxx
 *
 * \brief Implementation of base type logic.
 */

namespace vistk
{

pipeline_exception
::pipeline_exception() throw()
  : std::exception()
  , m_what()
{
}

pipeline_exception
::~pipeline_exception() throw()
{
}

char const*
pipeline_exception
::what() const throw()
{
  return m_what.c_str();
}

}
