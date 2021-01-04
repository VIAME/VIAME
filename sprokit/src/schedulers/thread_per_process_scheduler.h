// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_SCHEDULERS_SCHEDULERS_THREAD_PER_PROCESS_SCHEDULER_H
#define SPROKIT_SCHEDULERS_SCHEDULERS_THREAD_PER_PROCESS_SCHEDULER_H

#include <schedulers/schedulers_export.h>

#include <sprokit/pipeline/scheduler.h>

/**
 * \file thread_per_process_scheduler.h
 *
 * \brief Declaration of the thread-per-process scheduler.
 */

namespace sprokit
{

/**
 * \class thread_per_process_scheduler
 *
 * \brief A scheduler which runs each process in its own thread.
 *
 * \scheduler Run a thread for each process.
 */
class SCHEDULERS_NO_EXPORT thread_per_process_scheduler
  : public scheduler
{
  public:
    /**
     * \brief Constructor.
     *
     * \param pipe The pipeline to scheduler.
     * \param config Contains config for the scheduler.
     */
    thread_per_process_scheduler(pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config);

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
    std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_SCHEDULERS_SCHEDULERS_THREAD_PER_PROCESS_SCHEDULER_H
