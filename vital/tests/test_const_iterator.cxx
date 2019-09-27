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
// Test default construction - const
TEST( const_iterator, construct_default )
{
  vital::const_iterator<int> i;
}

// ----------------------------------------------------------------------------
// Test construction passing a generator function
TEST( const_iterator, construct_with_generator )
{
  using iter_t = vital::const_iterator<int>;

  iter_t::next_value_func_t i_gen = []()->iter_t::reference{
    static int v = 0;
    return v;
  };

  iter_t i( i_gen );
}

// ----------------------------------------------------------------------------
// Test copy construction
TEST( const_iterator, construct_copy )
{
  using iter_t = vital::const_iterator<int>;
  iter_t i1;
  iter_t i2( i1 );
}

// ----------------------------------------------------------------------------
// Test copy construction from non-const, past-end iterator of the same base
// type.
TEST( const_iterator, construct_copy_from_nonconst_pastend )
{
  vital::iterator<int> i1;
  vital::const_iterator<int> i2( i1 );
}

// ----------------------------------------------------------------------------
// Test copy construction fron non-const, not-past-end iterator of the same
// base type.
TEST( const_iterator, construct_copy_from_nonconst )
{
  // Silly next function that returns the same reference.
  vital::iterator<int> i1( []()->vital::iterator<int>::reference{
        static int v = 0;
        return v;
      });
  vital::const_iterator<int> i2( i1 );
  EXPECT_EQ( *i1, *i2 );
}

// ----------------------------------------------------------------------------
// Test value assignment
TEST( const_iterator, assignment )
{
  vital::const_iterator<int> i1;
  i1 = vital::const_iterator<int>();
  // Iteration type enforced at compile time. The following should fail:
  //i1 = vital::const_iterator<double>();
}

// ----------------------------------------------------------------------------
// Test prefix incrementing operator over integer sequence generator.
TEST( const_iterator, prefix_increment )
{
  using iter_t = vital::const_iterator<unsigned int>;

  unsigned int v = -1;
  iter_t::next_value_func_t nvf = [&]()->iter_t::reference{
    ++v;
    return v;
  };

  iter_t it( nvf );
  EXPECT_EQ( *it, 0 );
  EXPECT_EQ( v, 0 );

  EXPECT_EQ( *++it, 1 );
  EXPECT_EQ( v, 1 );
  EXPECT_EQ( *++it, 2 );
  EXPECT_EQ( v, 2 );
  EXPECT_EQ( *++it, 3 );
  EXPECT_EQ( v, 3 );
  EXPECT_EQ( *++it, 4 );
  EXPECT_EQ( v, 4 );
}

// ----------------------------------------------------------------------------
// Test postfix incrementing operator over integer sequence generator.
TEST( const_iterator, postfix_increment )
{
  using iter_t = vital::const_iterator<unsigned int>;

  // The postfix operation requires that the generator function return unique
  // references, so we make an array to iterate over.
  unsigned int a[] = {0, 1, 2, 3};
  iter_t::next_value_func_t nvf = [&]()->iter_t::reference{
    static size_t i = 0;
    return a[i++];
  };

  iter_t it( nvf );
  EXPECT_EQ( *it, 0 );
  EXPECT_EQ( *it++, 0 );
  EXPECT_EQ( *it++, 1 );
  EXPECT_EQ( *it++, 2 );
  EXPECT_EQ( *it, 3 );
}

// ----------------------------------------------------------------------------
// Test iterating over pointers and using arrow operator.
TEST( const_iterator, pointer_iteration_arrow_operator )
{
  // Simple wrapper structure to test arrow operations.
  struct int_container {
    int i;
    int_container( int v )
      : i( v )
    {}
  };
  using iter_t = vital::const_iterator< int_container >;

  int_container a[] = { int_container( 0 ),
                        int_container( 1 ),
                        int_container( 2 ) };
  iter_t it( [&] () ->iter_t::reference {
    static size_t i = 0;
    return a[i++];
  } );

  EXPECT_EQ( it->i, 0 );
  EXPECT_EQ( it++->i, 0 );
  EXPECT_EQ( it->i, 1 );
  EXPECT_EQ( (++it)->i, 2 );
}

// ----------------------------------------------------------------------------
// Test that two iterators are equal at points where their current values are
// equal or both are at the end of iteration because the generation function
// raise a stop iteration exception.
TEST( const_iterator, it_equality )
{
  using namespace std;

  using iter_t = vital::const_iterator<int>;
  int a[] = { 10, 11, 12, 13 };

  cout << "Creating iterators" << endl;
  iter_t it1( [&]()->iter_t::reference{
    static size_t i = 0;
    if( i == 4 )
    {
      cout << "Raising stop iteration" << endl;
      VITAL_THROW( vital::stop_iteration_exception, "test" );
    }
    cout << "returning a[" << i << "]" << endl;
    return a[i++];
  } );
  iter_t it2( [&]()->iter_t::reference{
    static size_t i = 0;
    if( i == 4 )
    {
      cout << "Raising stop iteration" << endl;
      VITAL_THROW( vital::stop_iteration_exception, "test" );
    }
    cout << "returning a[" << i << "]" << endl;
    return a[i++];
  } );

  cout << "testing first values" << endl;
  EXPECT_EQ( *it1, 10 );
  EXPECT_EQ( *it2, 10 );

  // Initial values, equal, iterators should be equal.
  EXPECT_TRUE( it1 == it2 );
  EXPECT_FALSE( it1 != it2 );

  // Move iterators out of sync
  ++it1;
  EXPECT_EQ( *it1, 11 );
  ++it1;
  EXPECT_EQ( *it1, 12 );

  ++it2;
  EXPECT_EQ( *it2, 11 );

  // Iterators should not be equal, current values not equal.
  EXPECT_FALSE( it1 == it2 );
  EXPECT_TRUE( it1 != it2 );

  // Move iterators to their end;
  ++it1; // now 13
  ++it1; // now end

  ++it2; // now 12
  ++it2; // now 13
  ++it2; // now end

  EXPECT_TRUE( it1 == it2 );
  EXPECT_FALSE( it1 != it2 );

  // Attempting to move past the end should yield more stop iteration.
  cout << "Atempting to iterate \"past end\"" << endl;
  ++it1;
  ++it1;
  ++it1;
  ++it1;

  // Iterators should both still be "past end" thus equal
  EXPECT_TRUE( it1 == it2 );
  EXPECT_FALSE( it1 != it2 );
}

// ----------------------------------------------------------------------------
// Test equality between const and non-const iterators
TEST( const_iterator, const_nonconst_equality )
{
  using iter_t       = vital::iterator<int>;
  using const_iter_t = vital::const_iterator<int>;

  iter_t it( []()->iter_t::reference{
        static int v = 0;
        ++v;
        if( v == 2 )
          VITAL_THROW( vital::stop_iteration_exception, "test" );
        return v;
      } );
  const_iter_t cit( []()->const_iter_t::reference{
        static int v = 0;
        ++v;
        if( v == 2 )
          VITAL_THROW( vital::stop_iteration_exception, "test" );
        return v;
      } );

  EXPECT_EQ( *it, 1 );
  EXPECT_EQ( *cit, 1 );
  EXPECT_EQ( it, cit );
  EXPECT_EQ( cit, it );

  ++it; // past end
  EXPECT_NE( it, cit );
  EXPECT_NE( cit, it );

  ++cit; // past end
  EXPECT_EQ( it, cit );
  EXPECT_EQ( cit, it );
}

// ----------------------------------------------------------------------------
// Test that an iterator given a function that immediately raises stop
// iteration is equivalent to a default constructed iterator which should
// represent an ended iterator.
TEST( const_iterator, immediate_stop_iteration )
{
  using test_iterator = vital::const_iterator<int>;

  // Next value function that immediately throws stop iteration.
  test_iterator::next_value_func_t stop_iter_func = []()->test_iterator::reference{
    VITAL_THROW( vital::stop_iteration_exception, "test" );
  };

  test_iterator it_empty( stop_iter_func );
  test_iterator it_end;

  EXPECT_EQ( it_empty, it_end );
}

// ----------------------------------------------------------------------------
// Test swap operation.
TEST( const_iterator, swap )
{
  using namespace std;

  using iter_t = vital::const_iterator<int>;

  // 1: 10:13
  iter_t it1( []()->iter_t::reference{
    static int a[] = { 10, 11, 12, 13 };
    static size_t i = 0;
    if( i == 4 )
    {
      cout << "Raising stop iteration" << endl;
      VITAL_THROW( vital::stop_iteration_exception, "test" );
    }
    cout << "returning a[" << i << "]" << endl;
    return a[i++];
  } );
  // 2: 20:23
  iter_t it2( []()->iter_t::reference{
    static int a[] = { 20, 21, 22, 23 };
    static size_t i = 0;
    if( i == 4 )
    {
      cout << "Raising stop iteration" << endl;
      VITAL_THROW( vital::stop_iteration_exception, "test" );
    }
    cout << "returning a[" << i << "]" << endl;
    return a[i++];
  } );

  EXPECT_EQ( *it1, 10 );
  EXPECT_EQ( *it2, 20 );

  swap( it1, it2 );
  EXPECT_EQ( *it1, 20 );
  EXPECT_EQ( *it2, 10 );

  // move the iterators a little.
  ++it1; // now 21
  ++it2; // now 11
  ++it2; // now 12
  swap( it1, it2 );
  EXPECT_EQ( *it1, 12 );
  EXPECT_EQ( *it2, 21 );

  // Move to end.
  ++it1; // now 13
  ++it1; // now end
  ++it2; // now 22
  ++it2; // now 23
  ++it2; // now end
  EXPECT_EQ( it1, it2 );
  swap( it1, it2 );
  EXPECT_EQ( it1, it2 );
}

// ----------------------------------------------------------------------------
// Test copying an iterator at different points of iteration to check that
// state is correctly transfered.
TEST( const_iterator, copy_during_iteration )
{
  using namespace std;

  using iter_t = vital::const_iterator<int>;

  // Values in range 10:13.
  iter_t it1( []()->iter_t::reference{
    static int a[] = { 10, 11, 12, 13 };
    static size_t i = 0;
    if( i == 4 )
    {
      cout << "Raising stop iteration" << endl;
      VITAL_THROW( vital::stop_iteration_exception, "test" );
    }
    cout << "returning a[" << i << "]" << endl;
    return a[i++];
  } );

  // end iter
  iter_t it_end;

  EXPECT_EQ( *it1, 10 );
  iter_t it2 = it1;
  EXPECT_EQ( *++it2, 11 );
  EXPECT_EQ( *++it2, 12 );
  iter_t it3 = it2;
  EXPECT_EQ( *++it3, 13 );
  EXPECT_EQ( ++it3, it_end );
  // NOTE: Previous iterator instances still dereference to their last value,
  //       as long as base array `a` is defined, however if any of them are
  //       incremented, since they all use the same next-value-generator
  //       function, they will be immediately a past-end iterator since the
  //       shared function now raises stop-iteration.
}

// ----------------------------------------------------------------------------
// Test an iteration with a representative "set" container.
//
class VectorIntSet
{
public:
  using const_iterator = vital::const_iterator< int >;

  VectorIntSet() = default;
  VectorIntSet( std::vector<int> iset )
    : m_vec( iset )
  {}

  ~VectorIntSet() = default;

  // Const iterator access
  const_iterator begin() const
  {
    return const_iterator( make_const_next_function() );
  }

  const_iterator end() const
  {
    return const_iterator();
  }

private:
  using vec_t = std::vector<int>;
  vec_t m_vec;

  const_iterator::next_value_func_t make_const_next_function() const
  {
    return [=] () ->const_iterator::reference {
      static vec_t::const_iterator cit = m_vec.begin();
      if( cit == m_vec.end() )
      {
        VITAL_THROW( vital::stop_iteration_exception, "test" );
      }
      return *(cit++);
    };
  }
};

TEST( const_iterator, example_set_const_iteration )
{
  std::vector<int> v;
  v.push_back(0);
  v.push_back(1);
  v.push_back(2);

  VectorIntSet vis( v );

  VectorIntSet::const_iterator it = vis.begin();
  EXPECT_EQ( *it, 0 );
  ++it;
  EXPECT_EQ( *it, 1 );
  ++it;
  EXPECT_EQ( *it, 2 );
  ++it;
  EXPECT_TRUE( it == vis.end() );
  EXPECT_EQ( it, vis.end() );
}
