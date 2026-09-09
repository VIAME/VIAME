// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief descriptor distance implementation

#include <viame/core_types/descriptor.h>

/// return the hamming_distance between two descriptors
int
kwiver::vital
::hamming_distance( vital::descriptor_sptr d1, vital::descriptor_sptr d2 )
{
  // Bit set count operation from
  // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

  auto const d1_bytes = d1->num_bytes();

  if( d1_bytes % 4 )
  {
    VITAL_THROW(
      vital::invalid_value,
      "Descriptor must be a multiple of four bytes long." );
  }

  auto const d2_bytes = d2->num_bytes();
  if( d1_bytes != d2_bytes )
  {
    VITAL_THROW(
      vital::invalid_value,
      "Descriptors must be the same number of bytes long" );
  }

  auto const num_ints_long( d1_bytes / 4 );

  auto pa = reinterpret_cast< uint32_t const* >( d1->as_bytes() );
  auto pb = reinterpret_cast< uint32_t const* >( d2->as_bytes() );

  size_t dist = 0;

  for( size_t i = 0; i < num_ints_long; i++, pa++, pb++ )
  {
    auto v = *pa ^ *pb;
    v = v - ( ( v >> 1 ) & 0x55555555 );
    v = ( v & 0x33333333 ) + ( ( v >> 2 ) & 0x33333333 );
    dist += ( ( ( v + ( v >> 4 ) ) & 0xF0F0F0F ) * 0x1010101 ) >> 24;
  }

  return dist;
}
