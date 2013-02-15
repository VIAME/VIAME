/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_GRAVL_CONFIG_H_
#define VISTK_PROCESSES_GRAVL_CONFIG_H_

#include <vistk/config.h>

/**
 * \file gravl-config.h
 *
 * \brief Defines for symbol visibility in the gravl processes.
 */

#ifndef VISTK_PROCESSES_GRAVL_EXPORT
#ifdef MAKE_VISTK_PROCESSES_GRAVL_LIB
/// Export the symbol if building the library.
#define VISTK_PROCESSES_GRAVL_EXPORT VISTK_EXPORT
#else
/// Import the symbol if including the library.
#define VISTK_PROCESSES_GRAVL_EXPORT VISTK_IMPORT
#endif // MAKE_VISTK_PROCESSES_GRAVL_LIB
/// Hide the symbol from the library interface.
#define VISTK_PROCESSES_GRAVL_NO_EXPORT VISTK_NO_EXPORT
#endif // VISTK_PROCESSES_GRAVL_EXPORT

#ifndef VISTK_PROCESSES_GRAVL_EXPORT_DEPRECATED
/// Mark as deprecated.
#define VISTK_PROCESSES_GRAVL_EXPORT_DEPRECATED VISTK_DEPRECATED VISTK_PROCESSES_GRAVL_EXPORT
#endif // VISTK_PROCESSES_GRAVL_EXPORT_DEPRECATED

#endif // VISTK_PROCESSES_GRAVL_CONFIG_H_
