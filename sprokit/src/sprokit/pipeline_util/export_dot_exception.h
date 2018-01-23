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

#ifndef SPROKIT_PIPELINE_UTIL_EXPORT_DOT_EXCEPTION_H
#define SPROKIT_PIPELINE_UTIL_EXPORT_DOT_EXCEPTION_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

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
    export_dot_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~export_dot_exception() noexcept;
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
    null_pipeline_export_dot_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~null_pipeline_export_dot_exception() noexcept;
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
    null_cluster_export_dot_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~null_cluster_export_dot_exception() noexcept;
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
    empty_name_export_dot_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~empty_name_export_dot_exception() noexcept;
};

}

#endif // SPROKIT_PIPELINE_UTIL_EXPORT_DOT_EXCEPTION_H
