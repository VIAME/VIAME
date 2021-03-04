// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ENUMERATE_MATRIX_H_
#define VITAL_ENUMERATE_MATRIX_H_

#include <Eigen/Sparse>

#include <memory>

namespace kwiver {
namespace vital {

/// Adaptor to iterate over non-zero cells in a sparse matrix
/**
 *  \sa enumerate
 */
template < typename ValueT, int Options, typename IndexT >
class sparse_matrix_enumerator
{
  public:
    class iterator;
    typedef iterator const_iterator;

    typedef typename Eigen::SparseMatrix< ValueT, Options, IndexT > matrix_t;
    typedef typename matrix_t::Index index_t;
    typedef typename matrix_t::InnerIterator iterator_t;

    sparse_matrix_enumerator(matrix_t const&);

    iterator begin() const;
    iterator end() const;

  protected:
    matrix_t const* m_matrix;
};

// ----------------------------------------------------------------------------
template < typename ValueT, int Options, typename IndexT >
class sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
{
  public:
    iterator();
    iterator(matrix_t const&);

    iterator(iterator const&);

    bool operator==(iterator const&) const;
    bool operator!=(iterator const& other) const { return !(*this == other); }

    iterator& operator++();

    iterator_t operator*() const { return *(this->m_inner); }

  protected:
    friend class sparse_matrix_enumerator;

    matrix_t const* m_matrix;
    index_t m_end;
    index_t m_outer;
    std::unique_ptr<iterator_t> m_inner;
};

// ----------------------------------------------------------------------------
template < typename ValueT, int Options, typename IndexT >
sparse_matrix_enumerator< ValueT, Options, IndexT >
::sparse_matrix_enumerator(matrix_t const& mat)
  : m_matrix(&mat)
{
}

// ----------------------------------------------------------------------------
template < typename ValueT, int Options, typename IndexT >
typename
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
sparse_matrix_enumerator< ValueT, Options, IndexT >
::begin() const
{
  return iterator(*(this->m_matrix));
}

// ----------------------------------------------------------------------------
template < typename ValueT, int Options, typename IndexT >
typename
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
sparse_matrix_enumerator< ValueT, Options, IndexT >
::end() const
{
  return iterator();
}

// ----------------------------------------------------------------------------
template < typename ValueT, int Options, typename IndexT >
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
::iterator()
  : m_matrix(0), m_end(0), m_outer(0)
{
}

template < typename ValueT, int Options, typename IndexT >
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
::iterator(Eigen::SparseMatrix< ValueT, Options, IndexT > const& mat)
  : m_matrix(&mat), m_end(mat.outerSize()), m_outer(0),
    m_inner(new iterator_t(mat, 0))
{
  if (!(*(this->m_inner)))
  {
    ++(*this);
  }
}

// ----------------------------------------------------------------------------
template < typename ValueT, int Options, typename IndexT >
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
::iterator(iterator const& other)
  : m_matrix(other.m_matrix), m_end(other.m_end), m_outer(other.m_outer),
    m_inner(other.m_inner ? new iterator_t(*other.m_inner) : 0)
{
}

  // ----------------------------------------------------------------------------
template < typename ValueT, int Options, typename IndexT >
bool
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
::operator==(iterator const& other) const
{
  auto const self_valid = !!this->m_inner;
  auto const other_valid = !!other.m_inner;

  if (!self_valid && !other_valid)
  {
    return true;
  }

  return (self_valid == other_valid &&
          this->m_matrix == other.m_matrix &&
          this->m_outer == other.m_outer &&
          this->m_inner->index() == other.m_inner->index());
}

  // ----------------------------------------------------------------------------
template < typename ValueT, int Options, typename IndexT >
typename
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator&
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
::operator++()
{
  if (this->m_inner && !(++(*(this->m_inner))))
  {
    if (++this->m_outer < this->m_end)
    {
      this->m_inner.reset(new iterator_t(*(this->m_matrix), m_outer));
      if (!(*(this->m_inner)))
      {
        return ++(*this);
      }
    }
    else
    {
      this->m_inner.reset();
    }
  }

  return *this;
}

// ----------------------------------------------------------------------------
/// Create adaptor to iterate over non-zero cells in a sparse matrix
/**
 *  This creates an adaptor () on an Eigen::SparseMatrix that can be iterated
 *  over with e.g. a range-based for loop in order to visit each non-zero cell
 *  in the matrix.
 */
template < typename ValueT, int Options, typename IndexT >
sparse_matrix_enumerator< ValueT, Options, IndexT > enumerate(
  Eigen::SparseMatrix< ValueT, Options, IndexT > const& mat)
{
  return sparse_matrix_enumerator< ValueT, Options, IndexT >(mat);
}

} } // end namespace

#endif // VITAL_ENUMERATE_MATRIX_H_
