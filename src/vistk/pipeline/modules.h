/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_MODULES_H
#define VISTK_PIPELINE_MODULES_H

#include "pipeline-config.h"

/**
 * \file modules.h
 *
 * \brief Functions dealing with modules in vistk.
 */

namespace vistk
{

/**
 * \brief Load modules from the system path.
 */
VISTK_PIPELINE_EXPORT void load_known_modules();

}

#endif // VISTK_PIPELINE_MODULES_H
