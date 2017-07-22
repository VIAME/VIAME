/*ckwg +29
 * Copyright 2012-2013 by Kitware, Inc.
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

#ifndef SPROKIT_PYTHON_UTIL_SET_INDEXING_SUITE_H
#define SPROKIT_PYTHON_UTIL_SET_INDEXING_SUITE_H

#include "python_gil.h"

#ifdef WIN32
#pragma warning (push)
#pragma warning (disable : 4244)
#pragma warning (disable : 4267)
#endif
#include <boost/python/suite/indexing/indexing_suite.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/lexical_cast.hpp>
#ifdef WIN32
#pragma warning (pop)
#endif

#include <algorithm>
#include <iterator>

namespace boost
{

namespace python
{

template <typename Container, bool NoProxy, typename DerivedPolicies>
class set_indexing_suite;

namespace detail
{

template <typename Container, bool NoProxy>
class final_set_derived_policies
  : public set_indexing_suite<Container, NoProxy, final_set_derived_policies<Container, NoProxy> >
{
};

}

template <
    typename Container
  , bool NoProxy = false
  , typename DerivedPolicies = detail::final_set_derived_policies<Container, NoProxy> >
class set_indexing_suite
  : public indexing_suite<Container, DerivedPolicies, NoProxy, true>
{
  public:
    typedef typename Container::value_type data_type;
    typedef typename Container::value_type key_type;
    typedef typename Container::size_type index_type;
    typedef typename Container::size_type size_type;
    typedef typename Container::difference_type difference_type;

    static bool
    contains(Container const& container, key_type const& key)
    {
      return container.count(key);
    }

    static index_type
    convert_index(Container const& /*container*/, PyObject* /*i*/)
    {
      index_error();

      // Prevent a warning.
      return index_type();
    }

    static void
    delete_item(Container& /*container*/, index_type /*i*/)
    {
      index_error();
    }

    static data_type
    get_item(Container const& /*container*/, index_type /*i*/)
    {
      index_error();

      return data_type();
    }

    static void
    set_item(Container& /*container*/, index_type /*i*/, data_type const& /*v*/)
    {
      index_error();
    }

    static size_t
    size(Container const& container)
    {
      return container.size();
    }

    template <typename Class>
    static void
    extension_def(Class& cl)
    {
      cl
        .def("__and__", &intersection)
        .def("__iand__", &intersection_update)
        .def("__ior__", &update)
        .def("__isub__", &difference_update)
        .def("__ixor__", &symmetric_difference_update)
        .def("__or__", &union_)
        .def("__rand__", &intersection)
        .def("__ror__", &union_)
        .def("__rsub__", &difference_reverse)
        .def("__rxor__", &symmetric_difference)
        .def("__sub__", &difference)
        .def("__xor__", &symmetric_difference)
        .def("add", &add)
        .def("clear", &clear)
        .def("copy", &copy)
        .def("difference", &difference)
        .def("difference_update", &difference_update)
        .def("discard", &discard)
        .def("intersection", &intersection)
        .def("intersection_update", &intersection_update)
        .def("isdisjoint", &isdisjoint)
        .def("issubset", &issubset)
        .def("issuperset", &issuperset)
        .def("pop", &pop)
        .def("remove", &remove)
        .def("symmetric_difference", &symmetric_difference)
        .def("symmetric_difference_update", &symmetric_difference_update)
        .def("union", &union_)
        .def("update", &update)
      ;
    }
  private:
    typedef typename Container::iterator iterator_type;
    typedef typename Container::const_iterator const_iterator_type;
    typedef std::insert_iterator<Container> output_iterator_type;

    static void
    add(Container& container, data_type const& v)
    {
      container.insert(v);
    }

    static void
    clear(Container& container)
    {
      container.clear();
    }

    static Container
    copy(Container const& container)
    {
      return Container(container.begin(), container.end());
    }

    static Container
    difference(Container const& container, Container const& other)
    {
      return set_operation(container, other, &set_difference_wrap<iterator_type, iterator_type, output_iterator_type>);
    }

    static Container
    difference_reverse(Container const& container, Container const& other)
    {
      return difference(other, container);
    }

    static void
    difference_update(Container& container, Container const& other)
    {
      container = difference(container, other);
    }

    static void
    discard(Container& container, data_type const& v)
    {
      container.erase(v);
    }

    static Container
    intersection(Container const& container, Container const& other)
    {
      return set_operation(container, other, &set_intersection_wrap<iterator_type, iterator_type, output_iterator_type>);
    }

    static void
    intersection_update(Container& container, Container const& other)
    {
      container = intersection(container, other);
    }

    static bool
    isdisjoint(Container const& container, Container const& other)
    {
      return intersection(container, other).empty();
    }

    static bool
    issubset(Container const& container, Container const& other)
    {
      return std::includes(other.begin(), other.end(),
                           container.begin(), container.end());
    }

    static bool
    issuperset(Container const& container, Container const& other)
    {
      return std::includes(container.begin(), container.end(),
                           other.begin(), other.end());
    }

    static void
    pop(Container& container)
    {
      if (container.empty())
      {
        sprokit::python::python_gil const gil;

        (void)gil;

        PyErr_SetString(PyExc_KeyError, "pop from an empty set");
        throw_error_already_set();
      }

      const_iterator_type i = container.begin();
      boost::random::mt19937 rng;
      boost::random::uniform_int_distribution<size_t> const diff(0, container.size() - 1);

      std::advance(i, diff(rng));

      container.erase(i);
    }

    static void
    remove(Container& container, data_type const& v)
    {
      const_iterator_type const i = container.find(v);

      if (i == container.end())
      {
        sprokit::python::python_gil const gil;

        (void)gil;

        PyErr_SetString(PyExc_KeyError, boost::lexical_cast<std::string>(v).c_str());
        throw_error_already_set();
      }

      container.erase(i);
    }

    static Container
    symmetric_difference(Container const& container, Container const& other)
    {
      return set_operation(container, other, &set_symmetric_difference_wrap<iterator_type, iterator_type, output_iterator_type>);
    }

    static void
    symmetric_difference_update(Container& container, Container const& other)
    {
      container = symmetric_difference(container, other);
    }

    static Container
    union_(Container const& container, Container const& other)
    {
      return set_operation(container, other, &merge_wrap<iterator_type, iterator_type, output_iterator_type>);
    }

    static void
    update(Container& container, Container const& other)
    {
      container = union_(container, other);
    }
  private:
    template <typename SetOperation>
    static Container
    set_operation(Container const& container, Container const& other, SetOperation operation)
    {
      Container result;

      output_iterator_type it(result, result.begin());

      operation(container.begin(), container.end(),
                other.begin(), other.end(),
                it);

      return result;
    }

    template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
    static OutputIterator
    set_difference_wrap(InputIterator1 first1, InputIterator1 last1,
                        InputIterator2 first2, InputIterator2 last2,
                        OutputIterator result)
    {
      return std::set_difference(first1, last1, first2, last2, result);
    }

    template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
    static OutputIterator
    set_intersection_wrap(InputIterator1 first1, InputIterator1 last1,
                          InputIterator2 first2, InputIterator2 last2,
                          OutputIterator result)
    {
      return std::set_intersection(first1, last1, first2, last2, result);
    }

    template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
    static OutputIterator
    merge_wrap(InputIterator1 first1, InputIterator1 last1,
               InputIterator2 first2, InputIterator2 last2,
               OutputIterator result)
    {
      return std::merge(first1, last1, first2, last2, result);
    }

    template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
    static OutputIterator
    set_symmetric_difference_wrap(InputIterator1 first1, InputIterator1 last1,
                                  InputIterator2 first2, InputIterator2 last2,
                                  OutputIterator result)
    {
      return std::set_symmetric_difference(first1, last1, first2, last2, result);
    }

    static void
    index_error()
    {
      sprokit::python::python_gil const gil;

      (void)gil;

      PyErr_SetString(PyExc_TypeError, "'set' object does not support indexing");
      throw_error_already_set();
    }
};

}

}

#endif // SPROKIT_PYTHON_UTIL_SET_INDEXING_SUITE_H
