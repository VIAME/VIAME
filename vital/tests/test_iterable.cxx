// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
