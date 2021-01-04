// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_SCHEDULERS_SYNC_SCHEDULER_H
#define SPROKIT_SCHEDULERS_SYNC_SCHEDULER_H

#include <schedulers/schedulers_export.h>

#include <sprokit/pipeline/scheduler.h>

/**
 * \file sync_scheduler.h
 *
 * \brief Declaration of the synchronized scheduler.
 */

namespace sprokit
{

/**
 * \class sync_scheduler
 *
 * \brief A scheduler which runs the entire pipeline in one thread.
 *
 * \scheduler Run the pipeline in one thread.
 */
class SCHEDULERS_NO_EXPORT sync_scheduler
  : public scheduler
{
  public:
    /**
     * \brief Constructor.
     *
     * \param pipe The pipeline to schedule.
     * \param config Contains config for the scheduler
     */
    sync_scheduler(pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config);

    /**
     * \brief Destructor.
     */
    ~sync_scheduler();

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
    std::unique_ptr<priv> d;
};

} // end namespace

#endif // SPROKIT_SCHEDULERS_SYNC_SCHEDULER_H
