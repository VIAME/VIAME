// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
#include <sprokit/pipeline_util/cluster_bakery.h>

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
     * \param type The type of the cluster.
     * \param description A description of the cluster.
     * \param ctor A function to create an instance of the cluster.
     */
    cluster_info(process::type_t const& type,
                 process::description_t const& description,
                 process_factory_func_t const& ctor );
    /**
     * \brief Destructor.
     */
    ~cluster_info() = default;

    /// The type of the cluster.
    process::type_t const type;

    /// A description of the cluster.
    process::description_t const description;

    /// A factory function to create an instance of the cluster.
    sprokit::process_factory_func_t const ctor;

    sprokit::cluster_bakery_sptr m_bakery;
};

/// A handle to information about a cluster.
using cluster_info_t =  std::shared_ptr<cluster_info>;

} // end namespace sprokit

#endif /* SPROKIT_PIPELINE_UTIL_CLUSTER_INFO_H */
