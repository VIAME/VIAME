/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

/**
 * \file
 * \brief C Interface to the Eigen matrix class
 */

#ifndef VITAL_C_EIGEN_H_
#define VITAL_C_EIGEN_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/error_handle.h>


/// Declare Eigen matrix interface functions for use with Vital
/**
 * \param T The data storage type like double or float
 * \param S The character suffix to use for naming of functions.
 * \param R Number of rows in the matrix. "Vector" types use this as the size
 *          parameter.
 * \param C Number of columns in the matrix. "Vector" types have a value of 1
 *          here.
 */
#define DECLARE_EIGEN_OPERATIONS( T, S, R, C ) \
/** Opaque Pointer Type */ \
typedef struct vital_eigen_matrix##R##x##C##S##_s vital_eigen_matrix##R##x##C##S##_t; \
\
/**
 * Create a new Eigen type-based Matrix of the given shape
 *
 * New matrices are column major in storage and uninitialized.
 */ \
VITAL_C_EXPORT \
vital_eigen_matrix##R##x##C##S##_t* \
vital_eigen_matrix##R##x##C##S##_new(); \
\
/**
 * Create a new Eigen type-based matrix with the given rows and columns.
 *
 * This is only useful for dynamic-size matrices, as fixed sized matrices can
 * only take their size as valid parameters here
 */ \
VITAL_C_EXPORT \
vital_eigen_matrix##R##x##C##S##_t* \
vital_eigen_matrix##R##x##C##S##_new_sized( ptrdiff_t rows, ptrdiff_t cols ); \
\
/** Destroy a given Eigen matrix instance */ \
VITAL_C_EXPORT \
void \
vital_eigen_matrix##R##x##C##S##_destroy( vital_eigen_matrix##R##x##C##S##_t *m, \
                                          vital_error_handle_t *eh ); \
\
/**
 * Get the value at a location
 * \param[in] m Matrix instance to get the data of
 * \param[in] row The row of the value to access
 * \param[in] col The column of the value to access
 * \param[in,out] eh Vital C error handle structure
 * \returns The value at the position specified
 */ \
VITAL_C_EXPORT \
T \
vital_eigen_matrix##R##x##C##S##_get( vital_eigen_matrix##R##x##C##S##_t *m, \
                                      ptrdiff_t row, ptrdiff_t col, \
                                      vital_error_handle_t *eh ); \
\
/**
 * Set the value at a location
 *
 * \param[in] m Matrix instance to set the values of
 * \param[in] row The row of the value to set
 * \param[in] col The column of the value to set
 * \param[in] value The value to set
 * \param[in,out] eh Vital C error handle structure
 */ \
VITAL_C_EXPORT \
void \
vital_eigen_matrix##R##x##C##S##_set( vital_eigen_matrix##R##x##C##S##_t *m, \
                                      ptrdiff_t row, ptrdiff_t col, \
                                      T value, \
                                      vital_error_handle_t *eh ); \
\
/**
 * Get the number of rows in the matrix
 */ \
VITAL_C_EXPORT \
ptrdiff_t \
vital_eigen_matrix##R##x##C##S##_rows( vital_eigen_matrix##R##x##C##S##_t *m, \
                                       vital_error_handle_t *eh ); \
\
/**
 * Get the number of columns in the matrix
 */ \
VITAL_C_EXPORT \
ptrdiff_t \
vital_eigen_matrix##R##x##C##S##_cols( vital_eigen_matrix##R##x##C##S##_t *m, \
                                       vital_error_handle_t *eh ); \
\
/**
 * Get the pointer increment between two consecutive rows.
 */ \
VITAL_C_EXPORT \
ptrdiff_t \
vital_eigen_matrix##R##x##C##S##_row_stride( vital_eigen_matrix##R##x##C##S##_t *m, \
                                             vital_error_handle_t *eh ); \
\
/**
 * Get the pointer increment between two consecutive columns.
 */ \
VITAL_C_EXPORT \
ptrdiff_t \
vital_eigen_matrix##R##x##C##S##_col_stride( vital_eigen_matrix##R##x##C##S##_t *m, \
                                             vital_error_handle_t *eh ); \
\
/**
 * Get the pointer to the vector's data array
 */ \
VITAL_C_EXPORT \
T* \
vital_eigen_matrix##R##x##C##S##_data( vital_eigen_matrix##R##x##C##S##_t *m, \
                                       vital_error_handle_t *eh );


/// Declare operations for both combinations of X and Y
#define DECLARE_EIGEN_RECTANGLES( T, S, X, Y ) \
DECLARE_EIGEN_OPERATIONS( T, S, X, Y ) \
DECLARE_EIGEN_OPERATIONS( T, S, Y, X ) \

/// Declare operations for all shapes
/**
 * The use of `X` in the below macros refers to matrices that are "dynamic" in
 * size (Eigen's definition). This basically means that the matrix size is
 * determined at run-time instead of compile time. With types that include `X`
 * size dimensions, the "...new_sized" constructor function must be used in
 * order to create a non-zero sized dimension.
 *
 * \param T Data type
 * \param S Type suffix
 */
#define DECLARE_EIGEN_ALL_SHAPES( T, S ) \
/* Vector shapes */                      \
DECLARE_EIGEN_RECTANGLES( T, S, 2, 1 )   \
DECLARE_EIGEN_RECTANGLES( T, S, 3, 1 )   \
DECLARE_EIGEN_RECTANGLES( T, S, 4, 1 )   \
DECLARE_EIGEN_RECTANGLES( T, S, X, 1 )   \
/* Square shapes */                      \
DECLARE_EIGEN_OPERATIONS( T, S, 2, 2 )   \
DECLARE_EIGEN_OPERATIONS( T, S, 3, 3 )   \
DECLARE_EIGEN_OPERATIONS( T, S, 4, 4 )   \
DECLARE_EIGEN_OPERATIONS( T, S, X, X )   \
/* Other Rectangular shapes */           \
DECLARE_EIGEN_RECTANGLES( T, S, 3, 2 )   \
DECLARE_EIGEN_RECTANGLES( T, S, 4, 2 )   \
DECLARE_EIGEN_RECTANGLES( T, S, 4, 3 )


DECLARE_EIGEN_ALL_SHAPES( double, d )
DECLARE_EIGEN_ALL_SHAPES( float,  f )


#undef DECLARE_EIGEN_OPERATIONS
#undef DECLARE_EIGEN_RECTANGLES
#undef DECLARE_EIGEN_ALL_SHAPES


#ifdef __cplusplus
}
#endif

#endif //VITAL_C_EIGEN_H_
