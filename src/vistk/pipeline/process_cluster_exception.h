/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PIPELINE_PROCESS_CLUSTER_EXCEPTION_H
#define SPROKIT_PIPELINE_PROCESS_CLUSTER_EXCEPTION_H

#include "pipeline-config.h"

#include "process_exception.h"
#include "types.h"

#include <string>

/**
 * \file process_cluster_exception.h
 *
 * \brief Header for exceptions used within \link sprokit::process_cluster process clusters\endlink.
 */

namespace sprokit
{

/**
 * \class process_cluster_exception process_cluster_exception.h <sprokit/pipeline/process_cluster_exception.h>
 *
 * \brief The base class for special exceptions thrown from a \ref process_cluster.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT process_cluster_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     */
    process_cluster_exception() throw();
    /**
     * \brief Destructor.
     */
    virtual ~process_cluster_exception() throw();
};

/**
 * \class mapping_after_process_exception process_cluster_exception.h <sprokit/pipeline/process_cluster_exception.h>
 *
 * \brief Thrown when a configuration is mapped after the process has been created.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT mapping_after_process_exception
  : public process_cluster_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param key The name of the configuration.
     * \param mapped_name The name of the process.
     * \param mapped_key The name of the configuration.
     */
    mapping_after_process_exception(process::name_t const& name, config::key_t const& key, process::name_t const& mapped_name, config::key_t const& mapped_key) throw();
    /**
     * \brief Destructor.
     */
    ~mapping_after_process_exception() throw();

    /// The name of the \ref process_cluster the mapping occurred in.
    process::name_t const m_name;
    /// The key of the configuration on the cluster.
    config::key_t const m_key;
    /// The name of the \ref process which was being mapped to.
    process::name_t const m_mapped_name;
    /// The key of the configuration on the \ref process being mapped to.
    config::key_t const m_mapped_key;
};

/**
 * \class mapping_to_read_only_value_exception process_cluster_exception.h <sprokit/pipeline/process_cluster_exception.h>
 *
 * \brief Thrown when a configuration is mapped to a read-only value.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT mapping_to_read_only_value_exception
  : public process_cluster_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param key The name of the configuration.
     * \param mapped_name The name of the process.
     * \param mapped_key The name of the configuration.
     */
    mapping_to_read_only_value_exception(process::name_t const& name, config::key_t const& key, config::value_t const& value, process::name_t const& mapped_name, config::key_t const& mapped_key, config::value_t const& ro_value) throw();
    /**
     * \brief Destructor.
     */
    ~mapping_to_read_only_value_exception() throw();

    /// The name of the \ref process_cluster the mapping occurred in.
    process::name_t const m_name;
    /// The key of the configuration on the cluster.
    config::key_t const m_key;
    /// The value of the configuration on the cluster.
    config::value_t const m_value;
    /// The name of the \ref process which was being mapped to.
    process::name_t const m_mapped_name;
    /// The key of the configuration on the \ref process being mapped to.
    config::key_t const m_mapped_key;
    /// The value of the configuration given.
    config::value_t const m_ro_value;
};

}

#endif // SPROKIT_PIPELINE_PROCESS_CLUSTER_EXCEPTION_H
