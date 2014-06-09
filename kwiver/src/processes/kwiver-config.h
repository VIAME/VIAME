/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#ifndef _KWIVER_KIWVER_CONFIG_H_
#define _KWIVER_KIWVER_CONFIG_H_

#include <sprokit/config.h>

/**
 * \file kwiver-config.h
 *
 * \brief Defines for symbol visibility in kwiver processes.
 */

#ifndef KWIVER_PROCESSES_EXPORT
#ifdef MAKE_KWIVER_PROCESSES_LIB
/// Export the symbol if building the library.
#define KWIVER_PROCESSES_EXPORT SPROKIT_EXPORT
#else
/// Import the symbol if including the library.
#define KWIVER_PROCESSES_EXPORT SPROKIT_IMPORT
#endif // MAKE_KWIVER_PROCESSES_LIB
/// Hide the symbol from the library interface.
#define KWIVER_PROCESSES_NO_EXPORT SPROKIT_NO_EXPORT
#endif // KWIVER_PROCESSES_EXPORT

#ifndef KWIVER_PROCESSES_EXPORT_DEPRECATED
/// Mark as deprecated.
#define KWIVER_PROCESSES_EXPORT_DEPRECATED SPROKIT_DEPRECATED KWIVER_PROCESSES_EXPORT
#endif // KWIVER_PROCESSES_EXPORT_DEPRECATED

#endif /* _KWIVER_KIWVER_CONFIG_H_ */
