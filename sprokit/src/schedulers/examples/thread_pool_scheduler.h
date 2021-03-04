// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_SCHEDULERS_EXAMPLES_SCHEDULERS_THREAD_POOL_SCHEDULER_H
#define SPROKIT_SCHEDULERS_EXAMPLES_SCHEDULERS_THREAD_POOL_SCHEDULER_H

#include <schedulers/examples/schedulers_examples_export.h>

#include <sprokit/pipeline/scheduler.h>

#include <cstddef>

/**
 * \file thread_pool_scheduler.h
 *
 * \brief Declaration of the thread pool scheduler.
 */

namespace sprokit
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
class SCHEDULERS_EXAMPLES_NO_EXPORT thread_pool_scheduler
  : public scheduler
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the edge.
     * \param pipe The pipeline to scheduler.
     */
    thread_pool_scheduler(pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config);
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
    std::unique_ptr<priv> d;
};

} // end namespace

#endif // SPROKIT_SCHEDULERS_EXAMPLES_SCHEDULERS_THREAD_POOL_SCHEDULER_H
