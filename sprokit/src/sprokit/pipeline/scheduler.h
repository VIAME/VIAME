/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SPROKIT_PIPELINE_SCHEDULER_H
#define SPROKIT_PIPELINE_SCHEDULER_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include "types.h"
#include <vital/config/config_block.h>
#include <vital/logger/logger.h>
#include <vital/noncopyable.h>

/**
 * \file scheduler.h
 *
 * \brief Header for \link sprokit::scheduler schedulers\endlink.
 */

namespace sprokit {

/**
 * \class scheduler scheduler.h <sprokit/pipeline/scheduler.h>
 *
 * \brief The base class for execution strategies on a \ref pipeline.
 *
 * This class is the abstract base class for all pipeline
 * schedulers. All concrete schedulers must be derived from this
 * interface.
 *
 * \ingroup base_classes
 */
class SPROKIT_PIPELINE_EXPORT scheduler
  : private kwiver::vital::noncopyable
{
  public:
    /// The type of registry keys.
    typedef std::string type_t;
    /// Scheduler description
    typedef std::string description_t;

    /**
     * \brief Destructor.
     */
    virtual ~scheduler();

    /**
     * \brief Start execution.
     *
     * \throws restart_scheduler_exception Thrown when the scheduler was already started.
     */
    void start();

    /**
     * \brief Wait until execution is finished.
     *
     * \throws restart_scheduler_exception Thrown when the scheduler has not been started.
     */
    void wait();

    /**
     * \brief Pause execution.
     *
     * \throws pause_before_start_exception Thrown when the scheduler has not been started.
     */
    void pause();

    /**
     * \brief Resume execution.
     *
     * \throws resume_unpaused_scheduler_exception Thrown when the scheduler is not paused.
     */
    void resume();

    /**
     * \brief Stop execution of the pipeline.
     *
     * \throws stop_before_start_exception Thrown when the scheduler has not been started.
     */
    void stop();

  protected:
    /**
     * \brief Constructor.
     *
     * \param pipe The pipeline to run.
     * \param config Contains configuration for the scheduler.
     */
    scheduler(pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config);

    /**
     * \brief Start execution.
     *
     * \warning Implementations should *not* return if this fails to start the
     * pipeline. Exceptions should be thrown instead.
     */
    virtual void _start() = 0;

    /**
     * \brief Wait until execution is finished.
     */
    virtual void _wait() = 0;

    /**
     * \brief Pause execution.
     */
    virtual void _pause() = 0;

    /**
     * \brief Resume execution.
     */
    virtual void _resume() = 0;

    /**
     * \brief Stop execution of the pipeline.
     *
     * This method may return before all processes are stopped. Call
     * wait() to synchronize with the termination event.
     *
     * \warning Implementations should *not* return if they fail to
     * stop the pipeline. Exceptions should be thrown instead.
     */
    virtual void _stop() = 0;

    /**
     * \brief Shuts down the scheduler.
     *
     * \note This should generally be called from the destructor of any
     * scheduler implementations.
     */
    void shutdown();

    /**
     * \brief The pipeline that should be run.
     *
     * \returns The pipeline.
     */
    pipeline_t pipeline() const;

    /**
     * \brief Get logger handle
     *
     * \returns Logger handle so log messages can be generated.
     */
    kwiver::vital::logger_handle_t logger();

    // The logger handle
    kwiver::vital::logger_handle_t m_logger;

  private:

    class SPROKIT_PIPELINE_NO_EXPORT priv;
    std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PIPELINE_SCHEDULER_H
