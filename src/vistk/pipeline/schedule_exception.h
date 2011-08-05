/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
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
 * \class null_schedule_config_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when \c NULL \ref config is passed to a schedule.
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

    /**
     * \brief A description of the exception.
     *
     * \returns A string describing what went wrong.
     */
    char const* what() const throw();
  private:
    std::string m_what;
};

}

#endif // VISTK_PIPELINE_SCHEDULE_EXCEPTION_H
