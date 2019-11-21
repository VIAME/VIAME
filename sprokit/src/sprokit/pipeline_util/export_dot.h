/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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

#ifndef SPROKIT_PIPELINE_UTIL_EXPORT_DOT_H
#define SPROKIT_PIPELINE_UTIL_EXPORT_DOT_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include <sprokit/pipeline/types.h>

#include <iosfwd>
#include <string>

/**
 * \file export_dot.h
 *
 * \brief Functions export a dot file for a pipeline.
 */

namespace sprokit
{

/**
 * \brief Exports a dot graph for a pipeline.
 *
 * Outputs a DOT formatted graph which represents the pipeline's requested
 * layout.
 *
 * \throws null_pipeline_export_dot_exception Thrown when \p pipe is \c NULL.
 *
 * \param ostr The stream to export to.
 * \param pipe The pipeline to export.
 * \param graph_name The name of the graph.
 * \param link_prefix A prefix to link processes for documentation
 */
SPROKIT_PIPELINE_UTIL_EXPORT void export_dot(std::ostream& ostr,
                                             pipeline_t const& pipe,
                                             std::string const& graph_name,
                                             std::string const& link_prefix);

/**
 * \brief Exports a dot graph for a pipeline.
 *
 * Outputs a DOT formatted graph which represents the pipeline's requested
 * layout.
 *
 * \throws null_pipeline_export_dot_exception Thrown when \p pipe is \c NULL.
 *
 * \param ostr The stream to export to.
 * \param pipe The pipeline to export.
 * \param graph_name The name of the graph.
 */
SPROKIT_PIPELINE_UTIL_EXPORT void export_dot(std::ostream& ostr,
                                             pipeline_t const& pipe,
                                             std::string const& graph_name);

/**
 * \brief Exports a dot graph for a cluster.
 *
 * Outputs a DOT formatted graph which represents the cluster's requested
 * layout.
 *
 * \throws null_cluster_export_dot_exception Thrown when \p cluster is \c NULL.
 *
 * \param ostr The stream to export to.
 * \param cluster The cluster to export.
 * \param graph_name The name of the graph.
 */
SPROKIT_PIPELINE_UTIL_EXPORT void export_dot(std::ostream& ostr,
                                             process_cluster_t const& cluster,
                                             std::string const& graph_name);

}

#endif // SPROKIT_PIPELINE_UTIL_EXPORT_DOT_H
