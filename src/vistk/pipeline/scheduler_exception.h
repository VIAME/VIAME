/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_SCHEDULER_EXCEPTION_H
#define VISTK_PIPELINE_SCHEDULER_EXCEPTION_H

#include "pipeline-config.h"

#include "types.h"

#include <string>

/**
 * \file scheduler_exception.h
 *
 * \brief Header for exceptions used within \link vistk::scheduler schedulers\endlink.
 */

namespace vistk
{

/**
 * \class scheduler_exception scheduler_exception.h <vistk/pipeline/scheduler_exception.h>
 *
 * \brief The base class for all exceptions thrown from an \ref scheduler.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT scheduler_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    scheduler_exception() throw();
    /**
     * \brief Destructor.
     */
    virtual ~scheduler_exception() throw();
};

/**
 * \class incompatible_pipeline_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler cannot execute the given pipeline.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT incompatible_pipeline_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    incompatible_pipeline_exception(std::string const& reason) throw();
    /**
     * \brief Destructor.
     */
    ~incompatible_pipeline_exception() throw();

    /// The reason why the scheduler cannot run the given pipeline.
    std::string const m_reason;
};

/**
 * \class null_scheduler_config_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \c NULL \ref config is passed to a scheduler.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_scheduler_config_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_scheduler_config_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_scheduler_config_exception() throw();
};

/**
 * \class null_scheduler_pipeline_exception scheduler_exception.h <vistk/pipeline/scheduler_exception.h>
 *
 * \brief Thrown when \c NULL \ref pipeline is passed to a scheduler.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_scheduler_pipeline_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_scheduler_pipeline_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_scheduler_pipeline_exception() throw();
};

/**
 * \class restart_scheduler_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is started after it has already been started.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT restart_scheduler_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    restart_scheduler_exception() throw();
    /**
     * \brief Destructor.
     */
    ~restart_scheduler_exception() throw();
};

/**
 * \class wait_before_start_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is waited on before it is started.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT wait_before_start_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    wait_before_start_exception() throw();
    /**
     * \brief Destructor.
     */
    ~wait_before_start_exception() throw();
};

/**
 * \class pause_before_start_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is paused before it is started.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pause_before_start_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pause_before_start_exception() throw();
    /**
     * \brief Destructor.
     */
    ~pause_before_start_exception() throw();
};

/**
 * \class repause_scheduler_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is paused while it is already paused.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT repause_scheduler_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    repause_scheduler_exception() throw();
    /**
     * \brief Destructor.
     */
    ~repause_scheduler_exception() throw();
};

/**
 * \class resume_unpaused_scheduler_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when an unpaused scheduler is resumed.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT resume_unpaused_scheduler_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    resume_unpaused_scheduler_exception() throw();
    /**
     * \brief Destructor.
     */
    ~resume_unpaused_scheduler_exception() throw();
};

/**
 * \class stop_before_start_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a scheduler is stopped before it is started.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT stop_before_start_exception
  : public scheduler_exception
{
  public:
    /**
     * \brief Constructor.
     */
    stop_before_start_exception() throw();
    /**
     * \brief Destructor.
     */
    ~stop_before_start_exception() throw();
};

}

#endif // VISTK_PIPELINE_SCHEDULER_EXCEPTION_H
