/*ckwg +5
 * Copyright 2011, 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "types.h"

/**
 * \file types.cxx
 *
 * \brief Implementation of base type logic.
 */

namespace sprokit
{

pipeline_exception
::pipeline_exception() SPROKIT_NOTHROW
  : std::exception()
  , m_what()
{
}

pipeline_exception
::~pipeline_exception() SPROKIT_NOTHROW
{
}

char const*
pipeline_exception
::what() const SPROKIT_NOTHROW
{
  return m_what.c_str();
}

}
