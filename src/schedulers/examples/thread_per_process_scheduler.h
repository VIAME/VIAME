/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCHEDULERS_EXAMPLES_SCHEDULERS_THREAD_PER_PROCESS_SCHEDULER_H
#define VISTK_SCHEDULERS_EXAMPLES_SCHEDULERS_THREAD_PER_PROCESS_SCHEDULER_H

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
class VISTK_SCHEDULERS_EXAMPLES_NO_EXPORT thread_per_process_scheduler
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
    ~thread_per_process_scheduler();
  protected:
    /**
     * \brief Starts execution.
     */
    void _start();
    /**
     * \brief Waits until execution is finished.
     */
    void _wait();
    /**
     * \brief Pauses execution.
     */
    void _pause();
    /**
     * \brief Resumes execution.
     */
    void _resume();
    /**
     * \brief Stop execution of the pipeline.
     */
    void _stop();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_SCHEDULERS_EXAMPLES_SCHEDULERS_THREAD_PER_PROCESS_SCHEDULER_H
