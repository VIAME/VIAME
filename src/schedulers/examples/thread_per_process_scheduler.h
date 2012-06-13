/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_PER_PROCESS_SCHEDULE_H
#define VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_PER_PROCESS_SCHEDULE_H

#include "examples-config.h"

#include <vistk/pipeline/scheduler.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file thread_per_process_scheduler.h
 *
 * \brief Declaration of the thread-per-process scheduler.
 */

namespace vistk
{

/**
 * \class thread_per_process_scheduler
 *
 * \brief A scheduler which runs each process in its own thread.
 *
 * \scheduler Run a thread for each process.
 */
class VISTK_SCHEDULES_EXAMPLES_NO_EXPORT thread_per_process_scheduler
  : public scheduler
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the edge.
     * \param pipe The pipeline to scheduler.
     */
    thread_per_process_scheduler(pipeline_t const& pipe, config_t const& config);
    /**
     * \brief Destructor.
     */
    virtual ~thread_per_process_scheduler();
  protected:
    /**
     * \brief Starts execution.
     */
    virtual void _start();
    /**
     * \brief Waits until execution is finished.
     */
    virtual void _wait();
    /**
     * \brief Stop execution of the pipeline.
     */
    virtual void _stop();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_PER_PROCESS_SCHEDULE_H
