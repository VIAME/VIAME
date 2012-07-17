/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCHEDULERS_EXAMPLES_SCHEDULERS_THREAD_POOL_SCHEDULER_H
#define VISTK_SCHEDULERS_EXAMPLES_SCHEDULERS_THREAD_POOL_SCHEDULER_H

#include "examples-config.h"

#include <vistk/pipeline/scheduler.h>

#include <boost/scoped_ptr.hpp>

#include <cstddef>

/**
 * \file thread_pool_scheduler.h
 *
 * \brief Declaration of the thread pool scheduler.
 */

namespace vistk
{

/**
 * \class thread_pool_scheduler
 *
 * \brief A scheduler which process execution among a group of threads.
 *
 * \scheduler Manages execution using a set number of threads.
 *
 * \configs
 *
 * \config{num_threads} The number of threads to run. A setting of \c 0 means "auto".
 */
class VISTK_SCHEDULERS_EXAMPLES_NO_EXPORT thread_pool_scheduler
  : public scheduler
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the edge.
     * \param pipe The pipeline to scheduler.
     */
    thread_pool_scheduler(pipeline_t const& pipe, config_t const& config);
    /**
     * \brief Destructor.
     */
    ~thread_pool_scheduler();
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

#endif // VISTK_SCHEDULERS_EXAMPLES_SCHEDULERS_THREAD_POOL_SCHEDULER_H
