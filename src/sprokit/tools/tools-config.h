/*ckwg +29
 * Copyright 2012-2013 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SPROKIT_TOOLS_CONFIG_H_
#define SPROKIT_TOOLS_CONFIG_H_

#include <sprokit/config.h>

/**
 * \file scoring-config.h
 *
 * \brief Defines for symbol visibility in scoring.
 */

#ifndef SPROKIT_TOOLS_EXPORT
#ifdef MAKE_SPROKIT_TOOLS_LIB
/// Export the symbol if building the library.
#define SPROKIT_TOOLS_EXPORT SPROKIT_EXPORT
#else
/// Import the symbol if including the library.
#define SPROKIT_TOOLS_EXPORT SPROKIT_IMPORT
#endif // MAKE_SPROKIT_TOOLS_LIB
/// Hide the symbol from the library interface.
#define SPROKIT_TOOLS_NO_EXPORT SPROKIT_NO_EXPORT
#endif // SPROKIT_TOOLS_EXPORT

#ifndef SPROKIT_TOOLS_EXPORT_DEPRECATED
/// Mark as deprecated.
#define SPROKIT_TOOLS_EXPORT_DEPRECATED SPROKIT_DEPRECATED SPROKIT_TOOLS_EXPORT
#endif // SPROKIT_TOOLS_EXPORT_DEPRECATED

#endif // SPROKIT_TOOLS_CONFIG_H_
