/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \brief descriptor distance implementation
 */

#include <vital/types/descriptor.h>

/// return the hamming_distance between two descriptors
int kwiver::vital::hamming_distance(vital::descriptor_sptr d1, vital::descriptor_sptr d2)
{
  // Bit set count operation from
  // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

  const int d1_bytes = d1->num_bytes();

  if (d1_bytes % 4)
  {
    VITAL_THROW( vital::invalid_value,"Descriptor must be a multiple of four bytes long.");
  }

  const int d2_bytes = d2->num_bytes();
  if (d1_bytes != d2_bytes)
  {
    VITAL_THROW( vital::invalid_value,"Descriptors must be the same number of bytes long");
  }

  const int num_ints_long(d1_bytes / 4);

  const int *pa = reinterpret_cast<const int*>(d1->as_bytes());
  const int *pb = reinterpret_cast<const int*>(d2->as_bytes());

  int dist = 0;

  for (int i = 0; i < num_ints_long; i++, pa++, pb++)
  {
    unsigned  int v = *pa ^ *pb;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  return dist;

}
