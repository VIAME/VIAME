/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_POOL_SCHEDULE_H
#define VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_POOL_SCHEDULE_H

#include "examples-config.h"

#include <vistk/pipeline/schedule.h>

#include <boost/scoped_ptr.hpp>

#include <cstddef>

/**
 * \file thread_pool_schedule.h
 *
 * \brief Declaration of the thread pool schedule.
 */

namespace vistk
{

/**
 * \class thread_pool_schedule
 *
 * \brief A schedule which process execution among a group of threads.
 *
 * \schedule Manages execution using a set number of threads.
 *
 * \configs
 *
 * \config{num_threads} The number of threads to run. A setting of \c 0 means "auto".
 */
class VISTK_SCHEDULES_EXAMPLES_NO_EXPORT thread_pool_schedule
  : public schedule
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the edge.
     * \param pipe The pipeline to schedule.
     * \param num_threads The number of threads to use. 0 means the number of processors available.
     */
    thread_pool_schedule(config_t const& config, pipeline_t const& pipe);
    /**
     * \brief Destructor.
     */
    virtual ~thread_pool_schedule();

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

#endif // VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_POOL_SCHEDULE_H
