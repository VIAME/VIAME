/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTIL_EXPORT_DOT_EXCEPTION_H
#define VISTK_PIPELINE_UTIL_EXPORT_DOT_EXCEPTION_H

#include "pipeline_util-config.h"

#include <vistk/pipeline/types.h>

/**
 * \file export_dot_exception.h
 *
 * \brief Header for exceptions used when export a pipeline to dot.
 */

namespace vistk
{

/**
 * \class export_dot_exception export_dot_exception.h <vistk/pipeline_util/export_dot_exception.h>
 *
 * \brief The base class for all exceptions thrown when exporting a pipeline to dot.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_UTIL_EXPORT export_dot_exception
  : public pipeline_exception
{
};

/**
 * \class null_pipeline_export_dot_exception load_pipe_exception.h <vistk/pipeline_util/load_pipe_exception.h>
 *
 * \brief The exception thrown when a \c NULL pipeline is given to export.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_UTIL_EXPORT null_pipeline_export_dot_exception
  : public export_dot_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_pipeline_export_dot_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_pipeline_export_dot_exception() throw();
};

}

#endif // VISTK_PIPELINE_UTIL_EXPORT_DOT_EXCEPTION_H
