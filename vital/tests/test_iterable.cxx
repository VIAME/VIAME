/*ckwg +29
 * Copyright 2017, 2019 by Kitware, Inc.
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

#include <iostream>

#include <gtest/gtest.h>

#include <vital/iterator.h>

using namespace kwiver;

// ----------------------------------------------------------------------------
int main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
// Test iterable interface with range-based loop.
namespace {

// Simple iterable over a vector of integers.
class SimpleIterable
  : public kwiver::vital::iterable< int >
{
public:
  SimpleIterable( std::vector<int> v )
    : v_( v )
  {}
  ~SimpleIterable() = default;

protected:
  iterator::next_value_func_t get_iter_next_func()
  {
    std::vector<int>::iterator it = v_.begin();
    return [=] () mutable ->iterator::reference {
      if( it == v_.end() ) { VITAL_THROW( kwiver::vital::stop_iteration_exception, "test" ); }
      return *(it++);
    };
  }

  const_iterator::next_value_func_t get_const_iter_next_func() const
  {
    std::vector<int>::const_iterator it = v_.begin();
    return [=] () mutable ->const_iterator::reference {
      if( it == v_.end() ) { VITAL_THROW( kwiver::vital::stop_iteration_exception, "test" ); }
      return *(it++);
    };
  }

private:
  std::vector<int> v_;
};

}

TEST( iterable, range_based_loop )
{
  using namespace std;
  using namespace kwiver::vital;

  // Make a simple vector of ints
  vector<int> v;
  v.push_back(0);
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.push_back(4);

  SimpleIterable si( v );
  int i = 0;
  for( int const & j : si )
  {
    EXPECT_EQ( j, i );
    ++i;
  }
  // We should have exited when i == 5.
  EXPECT_EQ( i, 5 );
}
