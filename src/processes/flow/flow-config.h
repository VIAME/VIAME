/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_FLOW_CONFIG_H_
#define SPROKIT_PROCESSES_FLOW_CONFIG_H_

#include <sprokit/config.h>

/**
 * \file flow-config.h
 *
 * \brief Defines for symbol visibility in the flow processes.
 */

#ifndef SPROKIT_PROCESSES_FLOW_EXPORT
#ifdef MAKE_SPROKIT_PROCESSES_FLOW_LIB
/// Export the symbol if building the library.
#define SPROKIT_PROCESSES_FLOW_EXPORT SPROKIT_EXPORT
#else
/// Import the symbol if including the library.
#define SPROKIT_PROCESSES_FLOW_EXPORT SPROKIT_IMPORT
#endif // MAKE_SPROKIT_PROCESSES_FLOW_LIB
/// Hide the symbol from the library interface.
#define SPROKIT_PROCESSES_FLOW_NO_EXPORT SPROKIT_NO_EXPORT
#endif // SPROKIT_PROCESSES_FLOW_EXPORT

#ifndef SPROKIT_PROCESSES_FLOW_EXPORT_DEPRECATED
/// Mark as deprecated.
#define SPROKIT_PROCESSES_FLOW_EXPORT_DEPRECATED SPROKIT_DEPRECATED SPROKIT_PROCESSES_FLOW_EXPORT
#endif // SPROKIT_PROCESSES_FLOW_EXPORT_DEPRECATED

#endif // SPROKIT_PROCESSES_FLOW_CONFIG_H_
