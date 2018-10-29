/*ckwg +29
 * Copyright 2011-2018 by Kitware, Inc.
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

/**
 * \file types.h
 *
 * \brief Common types used in the pipeline library.
 */

#ifndef SPROKIT_PIPELINE_TYPES_H
#define SPROKIT_PIPELINE_TYPES_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include <vital/vital_config.h>
#include <vital/exceptions/base.h>
#include <memory>

/**
 * \brief The namespace for all sprokit-related symbols.
 */
namespace sprokit
{

/**
 * \defgroup base_classes Base classes for the pipeline.
 * \defgroup exceptions Exceptions thrown within the pipeline.
 */
/// The type of a module name.
typedef std::string module_t;

class datum;
/// A typedef used to handle \link datum edge data\endlink.
typedef std::shared_ptr<datum const> datum_t;

class edge;
/// A typedef used to handle \link edge edges\endlink.
typedef std::shared_ptr<edge> edge_t;

class pipeline;
/// A typedef used to handle \link pipeline pipelines\endlink.
typedef std::shared_ptr<pipeline> pipeline_t;

class process;
/// A typedef used to handle \link process processes\endlink.
typedef std::shared_ptr<process> process_t;

class process_cluster;
/// A typedef used to handle \link process_cluster process clusters\endlink.
typedef std::shared_ptr<process_cluster> process_cluster_t;

class scheduler;
/// A typedef used to handle \link scheduler schedulers\endlink.
typedef std::shared_ptr<scheduler> scheduler_t;

class stamp;
/// A typedef used to handle \link stamp stamps\endlink.
typedef std::shared_ptr<stamp const> stamp_t;

/**
 * \class pipeline_exception types.h <sprokit/pipeline/types.h>
 *
 * \brief The base of all exceptions thrown within the pipeline.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT pipeline_exception
  : public kwiver::vital::vital_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~pipeline_exception() noexcept;
};

}

#endif // SPROKIT_PIPELINE_TYPES_H
