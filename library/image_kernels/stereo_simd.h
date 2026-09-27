/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_KERNELS_STEREO_SIMD_H
#define VIAME_IMAGE_KERNELS_STEREO_SIMD_H
#include <image_kernels/gaussian_simd.h>
#include <algorithm>
namespace viame
{
namespace image_kernels
{
namespace detail
{
#ifdef VIAME_GAUSSIAN_AVX
__attribute__ ( ( target ( "avx2" ) ) ) inline int
stereo_cost_avx ( int *dst, int count, int u, int u0, int u1, int const *v,
                  int const *low, int const *high, int at, int shift )
{
  auto const reverse = _mm256_setr_epi32 ( 7, 6, 5, 4, 3, 2, 1, 0 );
  auto const zero = _mm256_setzero_si256 ();
  auto const a = _mm256_set1_epi32 ( u ), a0 = _mm256_set1_epi32 ( u0 ),
             a1 = _mm256_set1_epi32 ( u1 );
  auto const bits = _mm256_set1_epi32 ( shift );
  int d = 0;
  for ( ; d + 8 <= count; d += 8 )
  {
    auto const b = _mm256_permutevar8x32_epi32 (
        _mm256_loadu_si256 ( reinterpret_cast<__m256i const *> ( v + at - d - 7 ) ),
        reverse );
    auto const b0 = _mm256_permutevar8x32_epi32 (
        _mm256_loadu_si256 ( reinterpret_cast<__m256i const *> ( low + at - d - 7 ) ),
        reverse );
    auto const b1 = _mm256_permutevar8x32_epi32 (
        _mm256_loadu_si256 ( reinterpret_cast<__m256i const *> ( high + at - d - 7 ) ),
        reverse );
    auto const c0 =
        _mm256_max_epi32 ( zero, _mm256_max_epi32 ( _mm256_sub_epi32 ( a, b1 ),
                                                    _mm256_sub_epi32 ( b0, a ) ) );
    auto const c1 =
        _mm256_max_epi32 ( zero, _mm256_max_epi32 ( _mm256_sub_epi32 ( b, a1 ),
                                                    _mm256_sub_epi32 ( a0, b ) ) );
    auto const cost = _mm256_srav_epi32 ( _mm256_min_epi32 ( c0, c1 ), bits );
    _mm256_storeu_si256 (
        reinterpret_cast<__m256i *> ( dst + d ),
        _mm256_add_epi32 (
            _mm256_loadu_si256 ( reinterpret_cast<__m256i const *> ( dst + d ) ),
            cost ) );
  }
  return d;
}
#endif
inline void stereo_cost ( int *dst, int count, int u, int u0, int u1, int const *v,
                          int const *low, int const *high, int at, int shift )
{
  int d = 0;
#ifdef VIAME_GAUSSIAN_AVX
  if ( gaussian_use_avx () )
  {
    d = stereo_cost_avx ( dst, count, u, u0, u1, v, low, high, at, shift );
  }
#endif
  for ( ; d < count; ++d )
  {
    auto const c0 = std::max ( 0, std::max ( u - high[at - d], low[at - d] - u ) );
    auto const c1 = std::max ( 0, std::max ( v[at - d] - u1, u0 - v[at - d] ) );
    dst[d] += std::min ( c0, c1 ) >> shift;
  }
}
} // namespace detail
} // namespace image_kernels
} // namespace viame
#endif
