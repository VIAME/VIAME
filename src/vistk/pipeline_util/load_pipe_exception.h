/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H
#define VISTK_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H

#include "pipeline_util-config.h"

#include <vistk/pipeline/types.h>

#include <string>

/**
 * \file load_pipe_exception.h
 *
 * \brief Header for exceptions used when loading a pipe declaration.
 */

namespace vistk
{

/**
 * \class load_pipe_exception load_pipe_exception.h <vistk/pipeline_util/load_pipe_exception.h>
 *
 * \brief The base class for all exceptions thrown when loading a pipe declaration.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_UTIL_EXPORT load_pipe_exception
  : public pipeline_exception
{
};

} // end namespace vistk

#endif // VISTK_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H
