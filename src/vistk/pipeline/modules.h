/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_MODULES_H
#define VISTK_PIPELINE_MODULES_H

#include "pipeline-config.h"

namespace vistk
{

/**
 * \brief Loads modules from the system path.
 */
void VISTK_PIPELINE_EXPORT load_known_modules();

} // end namespace vistk

#endif // VISTK_PIPELINE_MODULES_H
