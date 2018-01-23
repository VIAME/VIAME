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

#ifndef SPROKIT_PIPELINE_SCHEDULER_EXCEPTION_H
#define SPROKIT_PIPELINE_SCHEDULER_EXCEPTION_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include "types.h"

#include <string>

/**
 * \file scheduler_exception.h
 *
 * \brief Header for exceptions used within \link sprokit::scheduler schedulers\endlink.
 */

namespace sprokit
{

/**
 * \class scheduler_exception scheduler_exception.h <sprokit/pipeline/scheduler_exception.h>
 *
 * \brief The base class for all exceptions thrown from an \ref scheduler.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT scheduler_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    scheduler_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~scheduler_exception() noexcept;
};

/**
 * \class incompatible_pipeline_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler cannot execute the given pipeline.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT incompatible_pipeline_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    incompatible_pipeline_exception(std::string const& reason) noexcept;
    /**
     * \brief Destructor.
     */
    ~incompatible_pipeline_exception() noexcept;

    /// The reason why the scheduler cannot run the given pipeline.
    std::string const m_reason;
};

/**
 * \class null_scheduler_config_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \c NULL \ref config is passed to a scheduler.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_scheduler_config_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_scheduler_config_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~null_scheduler_config_exception() noexcept;
};

/**
 * \class null_scheduler_pipeline_exception scheduler_exception.h <sprokit/pipeline/scheduler_exception.h>
 *
 * \brief Thrown when \c NULL \ref pipeline is passed to a scheduler.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_scheduler_pipeline_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_scheduler_pipeline_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~null_scheduler_pipeline_exception() noexcept;
};

/**
 * \class restart_scheduler_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is started after it has already been started.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT restart_scheduler_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    restart_scheduler_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~restart_scheduler_exception() noexcept;
};

/**
 * \class wait_before_start_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is waited on before it is started.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT wait_before_start_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    wait_before_start_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~wait_before_start_exception() noexcept;
};

/**
 * \class pause_before_start_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is paused before it is started.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT pause_before_start_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pause_before_start_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~pause_before_start_exception() noexcept;
};

/**
 * \class repause_scheduler_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is paused while it is already paused.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT repause_scheduler_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    repause_scheduler_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~repause_scheduler_exception() noexcept;
};

/**
 * \class resume_before_start_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is resumed before it is started.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT resume_before_start_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    resume_before_start_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~resume_before_start_exception() noexcept;
};

/**
 * \class resume_unpaused_scheduler_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when an unpaused scheduler is resumed.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT resume_unpaused_scheduler_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    resume_unpaused_scheduler_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~resume_unpaused_scheduler_exception() noexcept;
};

/**
 * \class stop_before_start_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is stopped before it is started.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT stop_before_start_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    stop_before_start_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~stop_before_start_exception() noexcept;
};

}

#endif // SPROKIT_PIPELINE_SCHEDULER_EXCEPTION_H
