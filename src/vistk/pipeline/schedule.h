/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_SCHEDULE_H
#define VISTK_PIPELINE_SCHEDULE_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

/**
 * \file schedule.h
 *
 * \brief Header for \link vistk::schedule schedules\endlink.
 */

namespace vistk
{

/**
 * \class schedule schedule.h <vistk/pipeline/schedule.h>
 *
 * \brief The base class for execution strategies on a \ref pipeline.
 *
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT schedule
  : boost::noncopyable
{
  public:
    /**
     * \brief Destructor.
     */
    virtual ~schedule();

    /**
     * \brief Starts execution.
     *
     * \warning Implementations should *not* return if this fails to start the
     * pipeline. Exceptions should be thrown instead.
     */
    virtual void start() = 0;
    /**
     * \brief Stop execution of the pipeline.
     *
     * \warning Implementations should *not* return if they fail to stop the
     * pipeline. Exceptions should be thrown instead.
     */
    virtual void stop() = 0;
  protected:
    /**
     * \brief Constructor.
     *
     * \param config Contains configuration for the edge.
     * \param pipe The pipeline to run.
     */
    schedule(config_t const& config, pipeline_t const& pipe);

    /**
     * \brief The pipeline that should be run.
     *
     * \returns The pipeline.
     */
    pipeline_t pipeline() const;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_SCHEDULE_H
