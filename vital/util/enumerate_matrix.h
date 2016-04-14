/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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

template < typename ValueT, int Options, typename IndexT >
sparse_matrix_enumerator< ValueT, Options, IndexT >
::sparse_matrix_enumerator(matrix_t const& mat)
  : m_matrix(&mat)
{
}

template < typename ValueT, int Options, typename IndexT >
typename
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
sparse_matrix_enumerator< ValueT, Options, IndexT >
::begin() const
{
  return iterator(*(this->m_matrix));
}

template < typename ValueT, int Options, typename IndexT >
typename
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
sparse_matrix_enumerator< ValueT, Options, IndexT >
::end() const
{
  return iterator();
}

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

template < typename ValueT, int Options, typename IndexT >
sparse_matrix_enumerator< ValueT, Options, IndexT >::iterator
::iterator(iterator const& other)
  : m_matrix(other.m_matrix), m_end(other.m_end), m_outer(other.m_outer),
    m_inner(other.m_inner ? new iterator_t(*other.m_inner) : 0)
{
}

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

/// Create adaptor to iterate over non-zero cells in a sparse matrix
/**
 *  This creates an adaptor () on an Eigen::SparseMatrix that can be iterated
 *  over with e.g. VITAL_FOREACH in order to visit each non-zero cell in the
 *  matrix.
 */
template < typename ValueT, int Options, typename IndexT >
sparse_matrix_enumerator< ValueT, Options, IndexT > enumerate(
  Eigen::SparseMatrix< ValueT, Options, IndexT > const& mat)
{
  return sparse_matrix_enumerator< ValueT, Options, IndexT >(mat);
}

} } // end namespace


#endif // MAPTK_MATCH_MATRIX_H_
