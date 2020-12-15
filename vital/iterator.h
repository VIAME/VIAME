// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Vital templated iterator class.
 */

#ifndef KWIVER_VITAL_ITERATOR_H_
#define KWIVER_VITAL_ITERATOR_H_

#include <cstddef>
#include <functional>

#include <vital/exceptions/iteration.h>

namespace kwiver {
namespace vital {

// ============================================================================
/**
 * Internal base class for vital iterator types, fulfilling the input-iterator
 * concept.
 *
 * \tparam T Value type being iterated over. This should not be const.
 * \tparam Tb Base type for references and pointers. This will will be the
 *            same as T for a non-const iterator and the const version of T
 *            for a const iterator.
 */
template< typename T, typename Tb >
class base_iterator
{
public:
  using difference_type   = ptrdiff_t;
  using value_type        = T;
  using pointer           = Tb *;
  using reference         = Tb &;
  using iterator_category = std::input_iterator_tag;

  // Function that returns the reference to the next value to be iterated, or
  // throws a stop_iteration_exception to signal that iteration has ended.
  using next_value_func_t = std::function<reference()>;

  /**
   * Default destructor.
   */
  virtual ~base_iterator() = default;

  // Operator Overloads -------------------------------------------------------

  /**
   * Assignment operator overload
   *
   * Assigning another iterator to this iterator copies the other iterator's
   * next value function, current value pointer and past-end state flag.
   * Remember that copying the other iterator's next-value function does not
   * copy that function's state and iterations by the other iterator now affect
   * iteration returns of this iterator.
   *
   * \param rhs Other iterator instance to assign to this iterator instance.
   * \returns Reference to this iterator instance.
   */
  virtual
  base_iterator<T, Tb>& operator=( base_iterator<T, Tb> const & rhs )
  {
    this->m_next_value_func = rhs.m_next_value_func;
    this->m_cur_val_ptr = rhs.m_cur_val_ptr;
    this->m_at_end = rhs.m_at_end;
    return *this;
  }

  /**
   * Prefix increment operator overload (e.g. ++i)
   *
   * This requests and stores the reference to the next value provided by
   * the generation function.
   *
   * \returns Reference to this iterator instance.
   */
  virtual
  base_iterator<T, Tb>& operator++()
  {
    set_next_value();
    return *this;
  }
  /**
   * Postfix increment operator overload (e.g. i++).
   *
   * \returns A copy of this iterator before requesting the next value
   *          reference. If the next value function does not return
   *          sequentially unique references, then the returned iterator's
   *          reference will point to the same value as this iterator.
   */
  virtual
  base_iterator<T, Tb> operator++(int)
  {
    base_iterator<T, Tb> it(*this);
    set_next_value();
    return it;
  }

  /**
   * Dereference operator overload
   *
   * \returns A copy of the current iteration reference.
   */
  virtual
  value_type operator*() const
  {
    // NOTE: A forward-iterator sub-class will want to override this operator
    //       to return `reference`.
    return *m_cur_val_ptr;
  }

  /**
   * Arrow operator overload
   *
   * \returns Pointer to the current iteration reference.
   */
  virtual
  pointer operator->() const
  {
    return m_cur_val_ptr;
  }

  /**
   * Equality operator overload
   *
   * Two iterators are considered equal if their current iteration references
   * are equal in value or both iterators in a past-end state. If one
   * iterator is in a past-end state and the other is not, they two iterators
   * will never be equal.
   *
   * \returns If the two iterators are equal to each other.
   */
  friend bool operator==( base_iterator<T, Tb> const & lhs,
                          base_iterator<T, Tb> const & rhs )
  {
    // If both iterators are at their ends, they are equal.
    if( lhs.m_at_end && rhs.m_at_end )
    {
      return true;
    }
    // Both are not at their end, but one is: not equal.
    else if( lhs.m_at_end || rhs.m_at_end )
    {
      return false;
    }
    // Otherwise, the dereferenced value for each base_iterator are equal
    else
    {
      return (*lhs.m_cur_val_ptr) == (*rhs.m_cur_val_ptr);
    }
  }

  /**
   * Inequality operator overload
   *
   * \returns If the two iterators is *NOT* equal to each other.
   */
  friend bool operator!=( base_iterator<T, Tb> const & lhs,
                          base_iterator<T, Tb> const & rhs )
  {
    return ! (lhs == rhs);
  }

  /**
   * Override of swap function between two same-type iterators.
   *
   * Swaps function reference, current value pointer and past-end state
   * boolean.
   */
  friend void swap( base_iterator<T, Tb> const & lhs,
                    base_iterator<T, Tb> const & rhs )
  {
    next_value_func_t lhs_nvf = lhs.m_next_value_func;
    pointer           lhs_ref = lhs.m_cur_val_ptr;
    bool              lhs_ae  = lhs.m_at_end;
    lhs = rhs;
    rhs.m_next_value_func = lhs_nvf;
    rhs.m_cur_val_ptr     = lhs_ref;
    rhs.m_at_end          = lhs_ae;
  }

protected:
  // Next value reference generation function.
  next_value_func_t m_next_value_func;
  // Pointer to the most recent reference return from the generation function.
  pointer           m_cur_val_ptr;
  // If this iterator is past-end.
  bool              m_at_end;

  // Constructors: Protected to prevent construction of this base class -------
  /**
   * Construct default iterator.
   *
   * A default constructed iterator is set to be a past-end iterator and is
   * equal to any iterator past its end.
   */
  base_iterator()
    : m_next_value_func( nullptr )
    , m_cur_val_ptr( nullptr )
    , m_at_end( true )
  {}

  /**
   * Construct iterator with a function to yield the next iteration value
   * reference or raise a stop_iteration_exception.
   *
   * \param next_value_func Function that returns the reference to the next
   *                        value in a sequence until the end of that sequence,
   *                        then raising the ``stop_iteration_exception``.
   */
  base_iterator( next_value_func_t const & next_value_func )
    : m_next_value_func( next_value_func )
    , m_cur_val_ptr( nullptr )
    , m_at_end( false )
  {
    // Get the first iteration state value reference.
    set_next_value();
  }

  /**
   * Copy Constructor (shallow)
   *
   * Shallow copy iteration state from another iterator. This is shallow
   * because the next-value function's state is not fully copied, but now
   * shared between the original and copied iterator. Thus, as one iterator
   * is moved, the generation function state is changed, affecting subsequent
   * generator function returns in the other iterator.
   *
   * \param other Other iterator to copy from.
   */
  base_iterator( base_iterator<T, Tb> const & other )
    : m_next_value_func( other.m_next_value_func )
    , m_cur_val_ptr( other.m_cur_val_ptr )
    , m_at_end( other.m_at_end )
  {}

  /**
   * Acquire the next iteration value reference from the next value function.
   * Handles receiving a stop iteration exception from the function and flags
   * end of iteration if this occurs.
   */
  virtual
  void set_next_value()
  {
    try
    {
      m_cur_val_ptr = &m_next_value_func();
    }
    catch( stop_iteration_exception const & )
    {
      m_cur_val_ptr = nullptr;
      m_at_end = true;
    }
  }
};

// ============================================================================
// Forward declare const_iterator class for friending within iterator.
template< typename T >
class const_iterator;

/**
 * Vital templated non-const iterator class.
 *
 * \tparam T Value type being iterated over.
 */
template< typename T >
class iterator
  : public base_iterator< T, T >
{
public:
  using base_t = base_iterator< T, T >;

  // Expose iterator type definitions from base class.
  using difference_type   = typename base_t::difference_type;
  using value_type        = typename base_t::value_type;
  using pointer           = typename base_t::pointer;
  using reference         = typename base_t::reference;
  using iterator_category = typename base_t::iterator_category;

  using next_value_func_t = typename base_t::next_value_func_t;

  // Constructors -------------------------------------------------------------
  /**
   * Construct default iterator.
   *
   * A default constructed iterator is set to be a past-end iterator and is
   * equal to any iterator past its end.
   */
  iterator() = default;

  /**
   * Construct iterator with a function to yield the next iteration value
   * reference or raise a stop_iteration_exception.
   *
   * \param next_value_func Function that returns the reference to the next
   *                        value in a sequence until the end of that sequence,
   *                        then raising the ``stop_iteration_exception``.
   */
  iterator( next_value_func_t const & next_value_func )
    : base_t( next_value_func )
  {}

  /**
   * Copy Constructor (shallow)
   *
   * Shallow copy iteration state from another iterator. This is shallow
   * because the next-value function's state is not fully copied, but now
   * shared between the original and copied iterator. Thus, as one iterator
   * is moved, the generation function state is changed, affecting subsequent
   * generator function returns in the other iterator.
   *
   * \param other Other iterator to copy from.
   */
  iterator( iterator<T> const & other )
    : base_t( other )
  {}

  virtual ~iterator() = default;

  /// Use base class's equality operator overload.
  using base_t::operator=;

  /**
   * Allow const_iterator to access protected members when copy-constructing
   * from a non-const iterator instance.
   */
  friend class const_iterator<T>;
};

// ============================================================================
/**
 * Vital templated const iterator class.
 *
 * This class is similar to the vital::iterator except that returned references
 * and pointers from dereferencing and arrow operators are const.
 *
 * This subclass provides additional overloaded implementations of
 * ``friend bool operator==` and `friend bool operator!=` for the different
 * combinations of const and non-const iterator equality checks.
 *
 * \tparam T Value type being iterated over.
 */
template< typename T >
class const_iterator
  : public base_iterator< T, T const >
{
public:
  using base_t = base_iterator< T, T const >;

  // Expose iterator type definitions from base class.
  using difference_type   = typename base_t::difference_type;
  using value_type        = typename base_t::value_type;
  using pointer           = typename base_t::pointer;
  using reference         = typename base_t::reference;
  using iterator_category = typename base_t::iterator_category;

  using next_value_func_t = typename base_t::next_value_func_t;

  // Constructors -------------------------------------------------------------
  /**
   * Construct default iterator.
   *
   * A default constructed iterator is set to be a past-end iterator and is
   * equal to any iterator past its end.
   */
  const_iterator() = default;

  /**
   * Construct iterator with a function to yield the next iteration value
   * reference or raise a stop_iteration_exception.
   *
   * \param next_value_func Function that returns the reference to the next
   *                        value in a sequence until the end of that sequence,
   *                        then raising the ``stop_iteration_exception``.
   */
  const_iterator( next_value_func_t const & next_value_func )
    : base_t( next_value_func )
  {}

  /**
   * Copy Constructor (shallow)
   *
   * Shallow copy iteration state from another iterator. This is shallow
   * because the next-value function's state is not fully copied, but now
   * shared between the original and copied iterator. Thus, as one iterator
   * is moved, the generation function state is changed, affecting subsequent
   * generator function returns in the other iterator.
   *
   * \param other Other iterator to copy from.
   */
  const_iterator( const_iterator<T> const & other )
    : base_t( other )
  {}

  /**
   * Copy Constructor (shallow) from non-const iterator
   *
   * Shallow copy iteration state from another iterator. This is shallow
   * because the next-value function's state is not fully copied, but now
   * shared between the original and copied iterator. Thus, as one iterator
   * is moved, the generation function state is changed, affecting subsequent
   * generator function returns in the other iterator.
   *
   * \param other Other iterator to copy from.
   */
  const_iterator( iterator<T> const & other )
    : base_t()
  {
    this->m_next_value_func = other.m_next_value_func;
    this->m_cur_val_ptr = other.m_cur_val_ptr;
    this->m_at_end = other.m_at_end;
  }

  virtual ~const_iterator() = default;

  /// Use base class's equality operator overload.
  using base_t::operator=;

  /**
   * Friend operator== overload for testing [non-const, const] equality.
   */
  friend bool operator==( iterator<T> const & it,
                          const_iterator<T> const & cit )
  {
    return const_iterator<T>( it ) == cit;
  }
  /**
   * Friend operator== overload for testing [const, non-const] equality.
   */
  friend bool operator==( const_iterator<T> const & cit,
                          iterator<T> const & it )
  {
    return it == cit;
  }
  /**
   * Friend operator!= overload for testing [non-const, const] non-equality.
   */
  friend bool operator!=( iterator<T> const & it,
                          const_iterator<T> const & cit )
  {
    return ! (it == cit);
  }
  /**
   * Friend operator!= overload for testing [const, non-const] non-equality.
   */
  friend bool operator!=( const_iterator<T> const & cit,
                          iterator<T> const & it )
  {
    return ! (it == cit);
  }
};

// ============================================================================
/**
 * Pure-virtual mixin class to add iteration support to a class.
 *
 * \tparam Type to iterate over.
 */
template< typename T >
class iterable
{
public:
  using iterator       = vital::iterator< T >;
  using const_iterator = vital::const_iterator< T >;

  /// Constructor
  iterable() = default;
  /// Destructor
  virtual ~iterable() = default;

  /** @name Iterator Accessors
   * Accessors for const and non-const iterators
   */
  ///@{
  /**
   * Get the non-const iterator to the beginning of the collection.
   * @return An iterator over the objects in this collection.
   */
  virtual iterator begin()
  {
    return vital::iterator< T >( get_iter_next_func() );
  }

  /**
   * Get the non-const iterator past the end of the collection
   * @return An iterator base the end of this collection.
   */
  virtual iterator end()
  {
    return vital::iterator< T >();
  }

  /**
   * Get the const iterator to the beginning of the collection.
   * @return An iterator over the objects in this collection.
   */
  virtual const_iterator cbegin() const
  {
    return vital::const_iterator< T >( get_const_iter_next_func() );
  }

  /**
   * Get the const iterator past the end of the collection
   * @return An iterator base the end of this collection.
   */
  virtual const_iterator cend() const
  {
    return vital::const_iterator< T >();
  }
  ///@}

protected:
  /**
   * Get a new function that returns the sequence of values via subsequent
   * calls culminating with a stop_iteration_exception.
   *
   * @returns Function to generate value reference sequence.
   */
  virtual typename iterator::next_value_func_t
    get_iter_next_func() = 0;

  /**
   * Get a new function that returns the const sequence of values via
   * subsequent calls culminating with a stop_iteration_exception.
   *
   * @returns Function to generate value const-reference sequence.
   */
  virtual typename const_iterator::next_value_func_t
    get_const_iter_next_func() const = 0;
};

} } // end namespaces

#endif //KWIVER_VITAL_ITERATOR_H_
