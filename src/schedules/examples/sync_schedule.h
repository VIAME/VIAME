/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCHEDULES_EXAMPLES_SYNC_SCHEDULE_H
#define VISTK_SCHEDULES_EXAMPLES_SYNC_SCHEDULE_H

#include "examples-config.h"

#include <vistk/pipeline/schedule.h>

#include <boost/shared_ptr.hpp>

namespace vistk
{

/**
 * \class sync_schedule
 *
 * \brief A schedule which runs the entire pipeline in one thread.
 *
 * \schedule Runs the pipeline in one thread.
 *
 * \todo Does not handle infinite sources.
 */
class VISTK_SCHEDULES_EXAMPLES_NO_EXPORT sync_schedule
  : public schedule
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the edge.
     * \param pipe The pipeline to schedule.
     */
    sync_schedule(config_t const& config, pipeline_t const& pipe);
    /**
     * \brief Destructor.
     */
    virtual ~sync_schedule();

    /**
     * \brief Starts execution.
     */
    void start();
    /**
     * \brief Waits until execution is finished.
     */
    void wait();
    /**
     * \brief Stop execution of the pipeline.
     */
    void stop();
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_SCHEDULES_EXAMPLES_SYNC_SCHEDULE_H
