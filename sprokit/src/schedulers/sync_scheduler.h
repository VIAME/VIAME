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
