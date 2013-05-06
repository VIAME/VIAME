#if !defined(BOOST_PP_IS_ITERATING)

// Copyright David Abrahams 2002.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
# ifndef INVOKE_DWA20021122_HPP
#  define INVOKE_DWA20021122_HPP

#  include <boost/python/detail/prefix.hpp>
#  include <boost/python/detail/preprocessor.hpp>
#  include <boost/python/detail/none.hpp>

#  include <boost/type_traits/is_member_function_pointer.hpp>

#  include <boost/preprocessor/iterate.hpp>
#  include <boost/preprocessor/facilities/intercept.hpp>
#  include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#  include <boost/preprocessor/repetition/enum_trailing_binary_params.hpp>
#  include <boost/preprocessor/repetition/enum_binary_params.hpp>
#  include <boost/python/to_python_value.hpp>

#include <sprokit/python/util/python_allow_threads.h>
#include <boost/call_traits.hpp>
namespace boost { namespace python { namespace objects {
    template <class NextPolicies, class Iterator> struct iterator_range;
    namespace detail {
        template<class Target, class Iterator, class Accessor1, class Accessor2, class NextPolicies> struct py_iter_;
}}}}

// This file declares a series of overloaded invoke(...)  functions,
// used to invoke wrapped C++ function (object)s from Python. Each one
// accepts:
//
//   - a tag which identifies the invocation syntax (e.g. member
//   functions must be invoked with a different syntax from regular
//   functions)
//
//   - a pointer to a result converter type, used solely as a way of
//   transmitting the type of the result converter to the function (or
//   an int, if the return type is void).
//
//   - the "function", which may be a function object, a function or
//   member function pointer, or a defaulted_virtual_fn.
//
//   - The arg_from_python converters for each of the arguments to be
//   passed to the function being invoked.

namespace boost { namespace python { namespace detail {

// This "result converter" is really just used as a dispatch tag to
// invoke(...), selecting the appropriate implementation
typedef int void_result_to_python;

template <bool void_return, bool member>
struct invoke_tag_ {};

// A metafunction returning the appropriate tag type for invoking an
// object of type F with return type R.
template <class R, class F>
struct invoke_tag
  : invoke_tag_<
        is_same<R,void>::value
      , is_member_function_pointer<F>::value
    >
{
    typedef R result_type;
};

// Predeclaration of member && datum
template<class Data, class Class> struct member;
template <class Data> struct datum;

// Predeclaration of return extractor
template<class F> struct return_extract;
// Specialisation for member data & datum data
template<class Data, class Class>
struct return_extract<boost::python::detail::member<Data, Class> >
{
    typedef Data result_type;
    enum { GIL=true };
};
template<class Data>
struct return_extract<boost::python::detail::datum<Data> >
{
    typedef Data result_type;
    enum { GIL=true };
};
// Specialisation for iterators
template<class Target, class Iterator, class Accessor1, class Accessor2, class NextPolicies>
struct return_extract<boost::python::objects::detail::py_iter_<Target, Iterator, Accessor1, Accessor2, NextPolicies> >
{
    typedef boost::python::objects::iterator_range<NextPolicies, Iterator> result_type;
    enum { GIL=false };
};
// Fix for boost::python::objects::iterator_range<>::next (no other way of doing it :( )
template <class T> struct return_extract
{
    typedef typename T::result_type result_type;
    enum { GIL=false };
};

#  define BOOST_PP_ITERATION_PARAMS_1                                            \
        (3, (0, BOOST_PYTHON_MAX_ARITY, <boost/python/detail/invoke.hpp>))
#  include BOOST_PP_ITERATE()

}}} // namespace boost::python::detail

# endif // INVOKE_DWA20021122_HPP
#else

# define N BOOST_PP_ITERATION()

template<typename R, typename O BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
struct return_extract<R (O::*)(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, AC, ac) )>
{
    typedef R result_type;
    enum { GIL=true };
};

/*
 *template<typename R, typename O BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
 *struct return_extract<R (O::*)(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, AC, ac) ) throw()>
 *{
 *    typedef R result_type;
 *    enum { GIL=true };
 *};
 */

template<typename R, typename O BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
struct return_extract<R (O::*)(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, AC, ac) ) const>
{
    typedef R result_type;
    enum { GIL=true };
};

/*
 *template<typename R, typename O BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
 *struct return_extract<R (O::*)(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, AC, ac) ) const throw()>
 *{
 *    typedef R result_type;
 *    enum { GIL=true };
 *};
 */

template<typename R BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
struct return_extract<R (*)(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, AC, ac) )>
{
    typedef R result_type;
    enum { GIL=true };
};

/*
 *template<typename R BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
 *struct return_extract<R (*)(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, AC, ac) ) throw()>
 *{
 *    typedef R result_type;
 *    enum { GIL=true };
 *};
 */

#define CONVERT_ARG(z, n, _) typename boost::call_traits<typename AC##n::result_type>::param_type a##n = ac##n();

template <class RC, class F BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
inline PyObject* invoke(invoke_tag_<false,false> tag, RC const& rc, F& f BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, AC, & ac) )
{
    BOOST_PP_REPEAT(N, CONVERT_ARG, nil)

    typedef return_extract<F> return_info;

    sprokit::python::python_allow_threads allow(return_info::GIL);

    typename return_info::result_type ret = f( BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, a, BOOST_PP_INTERCEPT) );

    allow.release();

    return rc(ret);
}

template <class RC, class F BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
inline PyObject* invoke(invoke_tag_<true,false>, RC const&, F& f BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, AC, & ac) )
{
    BOOST_PP_REPEAT(N, CONVERT_ARG, nil)

    sprokit::python::python_allow_threads allow;

    f( BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, a, BOOST_PP_INTERCEPT) );

    allow.release();

    return none();
}

template <class RC, class F, class TC BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
inline PyObject* invoke(invoke_tag_<false,true> tag, RC const& rc, F& f, TC& tc BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, AC, & ac) )
{
    BOOST_PP_REPEAT(N, CONVERT_ARG, nil)

    typedef return_extract<F> return_info;

    sprokit::python::python_allow_threads allow(return_info::GIL);

    typename return_info::result_type ret = (tc().*f)(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, a, BOOST_PP_INTERCEPT));

    allow.release();

    return rc(ret);
}

template <class RC, class F, class TC BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
inline PyObject* invoke(invoke_tag_<true,true>, RC const&, F& f, TC& tc BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, AC, & ac) )
{
    BOOST_PP_REPEAT(N, CONVERT_ARG, nil)

    sprokit::python::python_allow_threads allow;

    (tc().*f)(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, a, BOOST_PP_INTERCEPT));

    allow.release();

    return none();
}

# undef CONVERT_ARG
# undef N

#endif // BOOST_PP_IS_ITERATING
