/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_DTOR_REGISTRY_EXCEPTION_H
#define VISTK_PIPELINE_DTOR_REGISTRY_EXCEPTION_H

#include "pipeline-config.h"

#include "types.h"

/**
 * \file dtor_registry_exception.h
 *
 * \brief Header for exceptions used within the \link vistk::dtor_registry process registry\endlink.
 */

namespace vistk
{

/**
 * \class dtor_registry_exception dtor_registry_exception.h <vistk/pipeline/dtor_registry_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref dtor_registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT dtor_registry_exception
  : public pipeline_exception
{
};

/**
 * \class null_dtor_exception dtor_registry_exception.h <vistk/pipeline/dtor_registry_exception.h>
 *
 * \brief Thrown when a \c NULL dtor function is added to the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_dtor_exception
  : public dtor_registry_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_dtor_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_dtor_exception() throw();
};

}

#endif // VISTK_PIPELINE_DTOR_REGISTRY_EXCEPTION_H
