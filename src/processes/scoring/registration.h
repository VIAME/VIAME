/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_SCORING_REGISTRATION_H
#define VISTK_PROCESSES_SCORING_REGISTRATION_H

#include "scoring-config.h"

/**
 * \file scoring/registration.h
 *
 * \brief Register processes for use.
 */

extern "C"
{

/**
 * \brief Register processes.
 */
VISTK_PROCESSES_SCORING_EXPORT void register_processes();

}

#endif // VISTK_PROCESSES_SCORING_REGISTRATION_H
