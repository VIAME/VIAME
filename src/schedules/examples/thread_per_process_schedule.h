/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_PER_PROCESS_SCHEDULE_H
#define VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_PER_PROCESS_SCHEDULE_H

#include "examples-config.h"

#include <vistk/pipeline/schedule.h>

#include <boost/shared_ptr.hpp>

namespace vistk
{

/**
 * \class thread_per_process_schedule
 *
 * \brief A schedule which runs each process in its own thread.
 *
 * \schedule Runs a thread for each process.
 */
class VISTK_SCHEDULES_EXAMPLES_NO_EXPORT thread_per_process_schedule
  : public schedule
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the edge.
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
     * \brief Stop execution of the pipeline.
     */
    virtual void stop();
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_SCHEDULES_EXAMPLES_SCHEDULES_THREAD_PER_PROCESS_SCHEDULE_H
