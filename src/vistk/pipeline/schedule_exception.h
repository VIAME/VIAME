/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_SCHEDULE_EXCEPTION_H
#define VISTK_PIPELINE_SCHEDULE_EXCEPTION_H

#include "pipeline-config.h"

#include "types.h"

#include <string>

/**
 * \file schedule_exception.h
 *
 * \brief Header for exceptions used within \link vistk::schedule schedules\endlink.
 */

namespace vistk
{

/**
 * \class schedule_exception schedule_exception.h <vistk/pipeline/schedule_exception.h>
 *
 * \brief The base class for all exceptions thrown from an \ref schedule.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT schedule_exception
  : public pipeline_exception
{
};

/**
 * \class incompatible_pipeline_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a schedule cannot execute the given pipeline.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT incompatible_pipeline_exception
  : public schedule_exception
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

    /// The reason why the schedule cannot run the given pipeline.
    std::string const m_reason;
};

/**
 * \class null_schedule_config_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \c NULL \ref config is passed to a schedule.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_schedule_config_exception
  : public schedule_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_schedule_config_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_schedule_config_exception() throw();
};

/**
 * \class null_schedule_pipeline_exception schedule_exception.h <vistk/pipeline/schedule_exception.h>
 *
 * \brief Thrown when \c NULL \ref pipeline is passed to a schedule.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_schedule_pipeline_exception
  : public schedule_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_schedule_pipeline_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_schedule_pipeline_exception() throw();
};

}

#endif // VISTK_PIPELINE_SCHEDULE_EXCEPTION_H
