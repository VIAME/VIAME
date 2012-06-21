/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_TYPE_MAPPINGS_H
#define VISTK_PYTHON_NUMPY_TYPE_MAPPINGS_H

#include <vil/vil_pixel_format.h>

/**
 * \file type_mappings.h
 *
 * \brief Macros to assist in converting between NumPy and vil images.
 */

#if __cplusplus >= 199711L
#include <complex>
#endif

#include <Python.h>

#include <numpy/arrayobject.h>

#if VXL_HAS_INT_64
#define INT64_CALLS(call, sep)             \
    SEP(sep) call(NPY_LONGLONG, long long) \
    SEP(sep) call(NPY_ULONGLONG, unsigned long long)
#else
#define INT64_CALLS(call, sep)
#endif

#if __cplusplus >= 199711L
/// \todo This is only guaranteed by C++11.
/// <http://stackoverflow.com/questions/5020076>
#define COMPLEX_CALLS(call, sep)                   \
    SEP(sep) call(NPY_CFLOAT, std::complex<float>) \
    SEP(sep) call(NPY_CDOUBLE, std::complex<double>)
#else
#define COMPLEX_CALLS(call, sep)
#endif

#define BEG(ctx) BEG_##ctx
#define SEP(ctx) SEP_##ctx
#define END(ctx) END_##ctx

#define SEMICOLON ;
#define BEG_LINES
#define SEP_LINES SEMICOLON
#define END_LINES SEMICOLON

#define BEG_NONE
#define SEP_NONE
#define END_NONE

#define FORMAT_CONVERSION(call, sep)               \
  BEG(sep) call(NPY_BOOL, bool)                    \
  SEP(sep) call(NPY_BYTE, signed char)             \
  SEP(sep) call(NPY_UBYTE, unsigned char)          \
  SEP(sep) call(NPY_SHORT, short)                  \
  SEP(sep) call(NPY_USHORT, unsigned short)        \
  SEP(sep) call(NPY_INT, int)                      \
  SEP(sep) call(NPY_UINT, unsigned int)            \
  SEP(sep) call(NPY_LONG, long)                    \
  SEP(sep) call(NPY_ULONG, unsigned long)          \
           INT64_CALLS(call, sep)                  \
  SEP(sep) call(NPY_FLOAT, float)                  \
  SEP(sep) call(NPY_DOUBLE, double)                \
           COMPLEX_CALLS(call, sep)                \
  END(sep)

#endif // VISTK_PYTHON_NUMPY_TYPE_MAPPINGS_H
