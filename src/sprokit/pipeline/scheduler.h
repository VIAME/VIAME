/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PIPELINE_SCHEDULER_H
#define SPROKIT_PIPELINE_SCHEDULER_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

/**
 * \file scheduler.h
 *
 * \brief Header for \link sprokit::scheduler schedulers\endlink.
 */

namespace sprokit
{

/**
 * \class scheduler scheduler.h <sprokit/pipeline/scheduler.h>
 *
 * \brief The base class for execution strategies on a \ref pipeline.
 *
 * \ingroup base_classes
 */
class SPROKIT_PIPELINE_EXPORT scheduler
  : boost::noncopyable
{
  public:
    /**
     * \brief Destructor.
     */
    virtual ~scheduler();

    /**
     * \brief Start execution.
     *
     * \throws restart_scheduler_exception Thrown when the scheduler was already started.
     */
    void start();
    /**
     * \brief Wait until execution is finished.
     *
     * \throws restart_scheduler_exception Thrown when the scheduler has not been started.
     */
    void wait();
    /**
     * \brief Pause execution.
     *
     * \throws pause_before_start_exception Thrown when the scheduler has not been started.
     */
    void pause();
    /**
     * \brief Resume execution.
     *
     * \throws resume_unpaused_scheduler_exception Thrown when the scheduler is not paused.
     */
    void resume();
    /**
     * \brief Stop execution of the pipeline.
     *
     * \throws stop_before_start_exception Thrown when the scheduler has not been started.
     */
    void stop();
  protected:
    /**
     * \brief Constructor.
     *
     * \param pipe The pipeline to run.
     * \param config Contains configuration for the edge.
     */
    scheduler(pipeline_t const& pipe, config_t const& config);

    /**
     * \brief Start execution.
     *
     * \warning Implementations should *not* return if this fails to start the
     * pipeline. Exceptions should be thrown instead.
     */
    virtual void _start() = 0;
    /**
     * \brief Wait until execution is finished.
     */
    virtual void _wait() = 0;
    /**
     * \brief Pause execution.
     */
    virtual void _pause() = 0;
    /**
     * \brief Resume execution.
     */
    virtual void _resume() = 0;
    /**
     * \brief Stop execution of the pipeline.
     *
     * \warning Implementations should *not* return if they fail to stop the
     * pipeline. Exceptions should be thrown instead.
     */
    virtual void _stop() = 0;

    /**
     * \brief Shuts down the scheduler.
     *
     * \note This should generally be called from the destructor of any
     * scheduler implementations.
     */
    void shutdown();

    /**
     * \brief The pipeline that should be run.
     *
     * \returns The pipeline.
     */
    pipeline_t pipeline() const;
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // SPROKIT_PIPELINE_SCHEDULER_H
