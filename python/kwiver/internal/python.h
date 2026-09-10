// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief Shim header to include instead of ``Python.h``.
 *
 * This handles an corner case when using MSVC and debug builds against a
 * release version on Python. See the note below for more details.
 */

#ifndef PYTHON_KWIVER_INTERNAL_PYTHON_H
#define PYTHON_KWIVER_INTERNAL_PYTHON_H

#if defined( _MSC_VER ) && defined( _DEBUG )
// Include these low level headers before undefing _DEBUG. Otherwise when doing
// a debug build against a release build of python the compiler will end up
// including these low level headers without DEBUG enabled, causing it to try
// and link release versions of this low level C api.
#include <assert.h>
#include <basetsd.h>
#include <ctype.h>
#include <errno.h>
#include <io.h>
#include <math.h>
#include <sal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <wchar.h>
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

#endif // PYTHON_KWIVER_INTERNAL_PYTHON_H
