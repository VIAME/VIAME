/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * @file   cluster_info.h
 * @brief  Interface to cluster info class.
 */

#ifndef SPROKIT_PIPELINE_UTIL_CLUSTER_INFO_H
#define SPROKIT_PIPELINE_UTIL_CLUSTER_INFO_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include "pipe_declaration_types.h"
#include <sprokit/pipeline/types.h>
#include <sprokit/pipeline/process_factory.h>


namespace sprokit {

// ------------------------------------------------------------------
/**
 * \class cluster_info pipe_bakery.h <sprokit/pipeline_util/pipe_bakery.h>
 *
 * \brief Information about a loaded cluster.
 */
class SPROKIT_PIPELINE_UTIL_EXPORT cluster_info
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type_ The type of the cluster.
     * \param description_ A description of the cluster.
     * \param ctor_ A function to create an instance of the cluster.
     */
    cluster_info(process::type_t const& type_,
                 process::description_t const& description_,
                 process_factory_func_t const& ctor_);
    /**
     * \brief Destructor.
     */
    ~cluster_info();

    /// The type of the cluster.
    process::type_t const type;

    /// A description of the cluster.
    process::description_t const description;

    /// A factory function to create an instance of the cluster.
    process_factory_func_t const ctor;
};

/// A handle to information about a cluster.
typedef std::shared_ptr<cluster_info> cluster_info_t;

} // end namespace sprokit

#endif /* SPROKIT_PIPELINE_UTIL_CLUSTER_INFO_H */
