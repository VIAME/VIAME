/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_FLOW_REGISTRATION_H
#define SPROKIT_PROCESSES_FLOW_REGISTRATION_H

#include "flow-config.h"

/**
 * \file flow/registration.h
 *
 * \brief Register processes for use.
 */

extern "C"
{

/**
 * \brief Register processes.
 */
SPROKIT_PROCESSES_FLOW_EXPORT void register_processes();

}

#endif // SPROKIT_PROCESSES_FLOW_REGISTRATION_H
