/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_PER_PROCESS_SCHEDULE_H
#define VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_PER_PROCESS_SCHEDULE_H

#include "examples-config.h"

#include <vistk/pipeline/schedule.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file thread_per_process_schedule.h
 *
 * \brief Declaration of the thread-per-process schedule.
 */

namespace vistk
{

/**
 * \class thread_per_process_schedule
 *
 * \brief A schedule which runs each process in its own thread.
 *
 * \schedule Runs a thread for each process.
 *
 * \todo Does not handle reentrant processes.
 */
class VISTK_SCHEDULES_EXAMPLES_NO_EXPORT thread_per_process_schedule
  : public schedule
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the edge.
     * \param pipe The pipeline to schedule.
     */
    thread_per_process_schedule(config_t const& config, pipeline_t const& pipe);
    /**
     * \brief Destructor.
     */
    virtual ~thread_per_process_schedule();

    /**
     * \brief Starts execution.
     */
    virtual void start();
    /**
     * \brief Waits until execution is finished.
     */
    virtual void wait();
    /**
     * \brief Stop execution of the pipeline.
     */
    virtual void stop();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_PER_PROCESS_SCHEDULE_H
