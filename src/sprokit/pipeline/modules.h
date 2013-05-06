/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PIPELINE_MODULES_H
#define SPROKIT_PIPELINE_MODULES_H

#include "pipeline-config.h"

/**
 * \file modules.h
 *
 * \brief Functions dealing with modules in sprokit.
 */

namespace sprokit
{

/**
 * \brief Load modules from the system path.
 */
SPROKIT_PIPELINE_EXPORT void load_known_modules();

}

#endif // SPROKIT_PIPELINE_MODULES_H
