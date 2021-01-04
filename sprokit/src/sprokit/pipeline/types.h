// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
