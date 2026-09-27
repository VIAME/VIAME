/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_KERNELS_GAUSSIAN_SIMD_H
#define VIAME_IMAGE_KERNELS_GAUSSIAN_SIMD_H
#include <cmath>
#include <cstddef>
#include <cstdlib>
#if ( defined( __GNUC__ ) || defined( __clang__ ) ) &&                                   \
    ( defined( __x86_64__ ) || defined( __i386__ ) )
#include <immintrin.h>
#define VIAME_GAUSSIAN_AVX 1
#endif
namespace viame
{
namespace image_kernels
{
namespace detail
{
inline bool gaussian_use_avx ()
{
#ifdef VIAME_GAUSSIAN_AVX
  static bool const supported = std::getenv ( "VIAME_DISABLE_SIMD" ) == nullptr &&
                                __builtin_cpu_supports ( "avx2" ) &&
                                __builtin_cpu_supports ( "fma" );
  return supported;
#else
  return false;
#endif
}
inline void gaussian_row_scalar ( float const *src, float *dst, std::size_t width,
                                  float const *tap, std::size_t n )
{
  for ( std::size_t x = 0; x < width; ++x )
  {
    float sum;
    if ( n == 3 )
      sum = std::fma ( src[x + 1], tap[1], ( src[x] + src[x + 2] ) * tap[2] );
    else if ( n == 5 )
      sum = std::fma (
          src[x + 4] + src[x], tap[4],
          std::fma ( src[x + 2], tap[2], ( src[x + 1] + src[x + 3] ) * tap[3] ) );
    else
    {
      sum = src[x] * tap[0];
      for ( std::size_t k = 1; k < n; ++k )
        sum = std::fma ( src[x + k], tap[k], sum );
    }
    dst[x] = sum;
  }
}
inline void gaussian_column_scalar ( float const *src, std::size_t width,
                                     std::size_t const *rows, float *dst,
                                     float const *tap, std::size_t n,
                                     std::size_t begin = 0 )
{
  auto const half = n / 2;
  for ( std::size_t x = begin; x < width; ++x )
  {
    auto sum = std::fma ( src[rows[half] * width + x], tap[half], 0.0f );
    for ( std::size_t k = 1; k <= half; ++k )
      sum = std::fma ( src[rows[half + k] * width + x] + src[rows[half - k] * width + x],
                       tap[half + k], sum );
    dst[x] = sum;
  }
}
#ifdef VIAME_GAUSSIAN_AVX
__attribute__ ( ( target ( "avx2,fma" ) ) ) inline void
gaussian_row_avx ( float const *src, float *dst, std::size_t width, float const *tap,
                   std::size_t n )
{
  std::size_t x = 0;
  for ( ; x + 8 <= width; x += 8 )
  {
    __m256 sum;
    if ( n == 3 )
      sum = _mm256_fmadd_ps (
          _mm256_loadu_ps ( src + x + 1 ), _mm256_set1_ps ( tap[1] ),
          _mm256_mul_ps ( _mm256_add_ps ( _mm256_loadu_ps ( src + x ),
                                          _mm256_loadu_ps ( src + x + 2 ) ),
                          _mm256_set1_ps ( tap[2] ) ) );
    else if ( n == 5 )
    {
      sum = _mm256_mul_ps ( _mm256_add_ps ( _mm256_loadu_ps ( src + x + 1 ),
                                            _mm256_loadu_ps ( src + x + 3 ) ),
                            _mm256_set1_ps ( tap[3] ) );
      sum = _mm256_fmadd_ps ( _mm256_loadu_ps ( src + x + 2 ), _mm256_set1_ps ( tap[2] ),
                              sum );
      sum = _mm256_fmadd_ps (
          _mm256_add_ps ( _mm256_loadu_ps ( src + x + 4 ), _mm256_loadu_ps ( src + x ) ),
          _mm256_set1_ps ( tap[4] ), sum );
    }
    else
    {
      sum = _mm256_mul_ps ( _mm256_loadu_ps ( src + x ), _mm256_set1_ps ( tap[0] ) );
      for ( std::size_t k = 1; k < n; ++k )
        sum = _mm256_fmadd_ps ( _mm256_loadu_ps ( src + x + k ),
                                _mm256_set1_ps ( tap[k] ), sum );
    }
    _mm256_storeu_ps ( dst + x, sum );
  }
  gaussian_row_scalar ( src + x, dst + x, width - x, tap, n );
}
__attribute__ ( ( target ( "avx2,fma" ) ) ) inline void
gaussian_column_avx ( float const *src, std::size_t width, std::size_t const *rows,
                      float *dst, float const *tap, std::size_t n )
{
  auto const half = n / 2;
  std::size_t x = 0;
  for ( ; x + 8 <= width; x += 8 )
  {
    auto sum = _mm256_fmadd_ps ( _mm256_loadu_ps ( src + rows[half] * width + x ),
                                 _mm256_set1_ps ( tap[half] ), _mm256_setzero_ps () );
    for ( std::size_t k = 1; k <= half; ++k )
      sum = _mm256_fmadd_ps (
          _mm256_add_ps ( _mm256_loadu_ps ( src + rows[half + k] * width + x ),
                          _mm256_loadu_ps ( src + rows[half - k] * width + x ) ),
          _mm256_set1_ps ( tap[half + k] ), sum );
    _mm256_storeu_ps ( dst + x, sum );
  }
  gaussian_column_scalar ( src, width, rows, dst, tap, n, x );
}
#endif
inline void gaussian_row ( float const *src, float *dst, std::size_t width,
                           float const *tap, std::size_t n )
{
#ifdef VIAME_GAUSSIAN_AVX
  if ( gaussian_use_avx () )
  {
    gaussian_row_avx ( src, dst, width, tap, n );
    return;
  }
#endif
  gaussian_row_scalar ( src, dst, width, tap, n );
}
inline void gaussian_column ( float const *src, std::size_t width,
                              std::size_t const *rows, float *dst, float const *tap,
                              std::size_t n )
{
#ifdef VIAME_GAUSSIAN_AVX
  if ( gaussian_use_avx () )
  {
    gaussian_column_avx ( src, width, rows, dst, tap, n );
    return;
  }
#endif
  gaussian_column_scalar ( src, width, rows, dst, tap, n );
}
} // namespace detail
} // namespace image_kernels
} // namespace viame
#endif
