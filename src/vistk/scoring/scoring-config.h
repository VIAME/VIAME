/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_SCORING_CONFIG_H_
#define SPROKIT_SCORING_CONFIG_H_

#include <sprokit/config.h>

/**
 * \file scoring-config.h
 *
 * \brief Defines for symbol visibility in scoring.
 */

#ifndef SPROKIT_SCORING_EXPORT
#ifdef MAKE_SPROKIT_SCORING_LIB
/// Export the symbol if building the library.
#define SPROKIT_SCORING_EXPORT SPROKIT_EXPORT
#else
/// Import the symbol if including the library.
#define SPROKIT_SCORING_EXPORT SPROKIT_IMPORT
#endif // MAKE_SPROKIT_SCORING_LIB
/// Hide the symbol from the library interface.
#define SPROKIT_SCORING_NO_EXPORT SPROKIT_NO_EXPORT
#endif // SPROKIT_SCORING_EXPORT

#ifndef SPROKIT_SCORING_EXPORT_DEPRECATED
/// Mark as deprecated.
#define SPROKIT_SCORING_EXPORT_DEPRECATED SPROKIT_DEPRECATED SPROKIT_SCORING_EXPORT
#endif // SPROKIT_SCORING_EXPORT_DEPRECATED

#endif // SPROKIT_SCORING_CONFIG_H_
