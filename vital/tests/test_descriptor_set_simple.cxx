/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Tests for simple_descriptor_set.
 */

#include <gtest/gtest.h>

#include <vital/types/descriptor_set.h>

// ----------------------------------------------------------------------------
int
main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
// Test construction of simple descriptor set.
TEST( descriptor_set_simple, construct_default )
{
  using namespace kwiver::vital;
  simple_descriptor_set ds;
}

// ----------------------------------------------------------------------------
// Test using range-based for-loop with empty dscriptor-set
TEST( descriptor_set_simple, range_based_loop_empty )
{
  using namespace kwiver::vital;

  // Range-based loop over empty set.
  simple_descriptor_set ds;
  int i = 0;
  for( descriptor_sptr const d : ds )
  {
    ++i;
  }
  EXPECT_EQ( i, 0 );
}

// ----------------------------------------------------------------------------
// Test using range-based for-loop with non-empty dscriptor-set
TEST( descriptor_set_simple, range_based_loop )
{
  using namespace kwiver::vital;

  // Make simple vector of descriptor_sptr
  std::vector< descriptor_sptr > dsptr_vec;
  for( int i=0; i < 3; ++i )
  {
    descriptor_fixed<int,2> *d = new descriptor_fixed<int,2>();
    d->raw_data()[0] = i;
    d->raw_data()[1] = i;
    dsptr_vec.push_back( descriptor_sptr( d ) );
  }

  // Range-based loop over empty set.
  simple_descriptor_set ds( dsptr_vec );
  size_t i = 0;
  for( descriptor_sptr const d : ds )
  {
    EXPECT_EQ( d->size(), 2 );
    EXPECT_EQ( d->as_double()[0], i );
    EXPECT_EQ( d->as_double()[1], i );
    ++i;
  }
  EXPECT_EQ( i, 3 );
}
