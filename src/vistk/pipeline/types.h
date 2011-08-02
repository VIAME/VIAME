/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_TYPES_H
#define VISTK_PIPELINE_TYPES_H

#include "pipeline-config.h"

#include <boost/shared_ptr.hpp>

#include <exception>

/**
 * \file types.h
 *
 * \brief Common types used in the pipeline library.
 */

/**
 * \brief The namespace for all vistk-related symbols.
 */
namespace vistk
{

/**
 * \defgroup base_classes Base classes for the pipeline.
 * \defgroup registries Registries of different types of pipeline objects.
 * \defgroup exceptions Exceptions thrown within the pipeline.
 */

class config;
/// A typedef used to handle \link config configurations\endlink.
typedef boost::shared_ptr<config> config_t;

class datum;
/// A typedef used to handle \link datum edge data\endlink.
typedef boost::shared_ptr<datum const> datum_t;

class edge;
/// A typedef used to handle \link edge edges\endlink.
typedef boost::shared_ptr<edge> edge_t;

class pipeline;
/// A typedef used to handle \link pipeline pipelines\endlink.
typedef boost::shared_ptr<pipeline> pipeline_t;

class process;
/// A typedef used to handle \link process processes\endlink.
typedef boost::shared_ptr<process> process_t;

class process_registry;
/// A typedef used to handle \link process_registry process registries\endlink.
typedef boost::shared_ptr<process_registry> process_registry_t;

class schedule;
/// A typedef used to handle \link schedule schedules\endlink.
typedef boost::shared_ptr<schedule> schedule_t;

class schedule_registry;
/// A typedef used to handle \link schedule_registry schedule registries\endlink.
typedef boost::shared_ptr<schedule_registry> schedule_registry_t;

class stamp;
/// A typedef used to handle \link stamp stamps\endlink.
typedef boost::shared_ptr<stamp const> stamp_t;

/**
 * \class pipeline_exception types.h <vistk/pipeline/types.h>
 *
 * \brief The base of all exceptions thrown within the pipeline.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pipeline_exception
  : public std::exception
{
};

}

#endif // VISTK_PIPELINE_TYPES_H
