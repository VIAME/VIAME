/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_MODULES_PYTHON_REGISTRATION_H
#define VISTK_MODULES_PYTHON_REGISTRATION_H

#include "modules-config.h"

/**
 * \file python/modules/registration.h
 *
 * \brief Register processes for use.
 */

extern "C"
{

/**
 * \brief Register processes.
 */
void VISTK_MODULES_PYTHON_EXPORT register_processes();
/**
 * \brief Register schedules.
 */
void VISTK_MODULES_PYTHON_EXPORT register_schedules();

}

#endif // VISTK_MODULES_PYTHON_REGISTRATION_H
