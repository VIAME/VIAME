/*ckwg +5
 * Copyright 2011, 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PIPELINE_UTIL_EXPORT_DOT_EXCEPTION_H
#define SPROKIT_PIPELINE_UTIL_EXPORT_DOT_EXCEPTION_H

#include "pipeline_util-config.h"

#include <sprokit/pipeline/types.h>

/**
 * \file export_dot_exception.h
 *
 * \brief Header for exceptions used when export a pipeline to dot.
 */

namespace sprokit
{

/**
 * \class export_dot_exception export_dot_exception.h <sprokit/pipeline_util/export_dot_exception.h>
 *
 * \brief The base class for all exceptions thrown when exporting a pipeline to dot.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT export_dot_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    export_dot_exception() throw();
    /**
     * \brief Destructor.
     */
    virtual ~export_dot_exception() throw();
};

/**
 * \class null_pipeline_export_dot_exception load_pipe_exception.h <sprokit/pipeline_util/load_pipe_exception.h>
 *
 * \brief The exception thrown when a \c NULL pipeline is given to export.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT null_pipeline_export_dot_exception
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

/**
 * \class null_cluster_export_dot_exception load_pipe_exception.h <sprokit/pipeline_util/load_pipe_exception.h>
 *
 * \brief The exception thrown when a \c NULL cluster is given to export.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT null_cluster_export_dot_exception
  : public export_dot_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_cluster_export_dot_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_cluster_export_dot_exception() throw();
};

/**
 * \class empty_name_export_dot_exception load_pipe_exception.h <sprokit/pipeline_util/load_pipe_exception.h>
 *
 * \brief The exception thrown when a process in a pipeline has an empty name.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT empty_name_export_dot_exception
  : public export_dot_exception
{
  public:
    /**
     * \brief Constructor.
     */
    empty_name_export_dot_exception() throw();
    /**
     * \brief Destructor.
     */
    ~empty_name_export_dot_exception() throw();
};

}

#endif // SPROKIT_PIPELINE_UTIL_EXPORT_DOT_EXCEPTION_H
