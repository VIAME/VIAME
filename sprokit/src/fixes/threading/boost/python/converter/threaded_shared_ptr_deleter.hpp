// Copyright David Abrahams 2002.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#ifndef THREADED_SHARED_PTR_DELETER_DWA2002121_HPP
# define THREADED_SHARED_PTR_DELETER_DWA2002121_HPP

#include <sprokit/python/util/python_gil.h>

# include <boost/python/converter/shared_ptr_deleter.hpp>

namespace boost { namespace python { namespace converter {

struct threaded_shared_ptr_deleter
    : public shared_ptr_deleter
{
    threaded_shared_ptr_deleter(handle<> owner)
        : shared_ptr_deleter(owner)
    {
    }
    ~threaded_shared_ptr_deleter()
    {
    }

    void operator()(void const* ptr)
    {
        sprokit::python::python_gil gil;
        (void)gil;

        shared_ptr_deleter::operator () (ptr);
    }
};

}}} // namespace boost::python::converter

#endif // THREADED_SHARED_PTR_DELETER_DWA2002121_HPP
