Common Structures
=================

.. _vital_iterator:

Iterators
---------
Iterators provide a container-independent way to access elements in an
aggregate structure without exposing or requiring any underlying structure.
This vital implementation intends to provide an easy way to create an input
iterator via a value reference generation function, similar to how iterators
in the python language function.

Each iterator class descends from a common base-class due to a wealth of shared
functionality. This base class has protected constructors in order to prevent
direct use.

It is currently undefined behavior when dereferencing an iterator after source
data has been released (e.g. if the ``next_value_func`` is iterating
over a vector of values and the source vector is released in the middle of
iteration).

Generation Function
^^^^^^^^^^^^^^^^^^^
The value reference generation function's purpose is to return the
reference to the next value of type T in a sequence every time it is called.
When the end of a specific is reached, an exception is raised to signal the
end of iteration.`

Upon incrementing this iterator, this ``next`` function is called and the
reference it returns is retained in order to yield the value or reference
when the ``*`` or ``->`` operator is invoked on this iterator.

*Generator function caveat*:
Since the next-value generator function is to return references, the
function should ideally return unique references every time it is called.
If this is not the case, the prefix-increment operator does not function
correctly since the returned iterator copy and old, incremented iterator
share the same reference and thus share the same value yielded.

Providing a function
""""""""""""""""""""
Next value functions can be provided in various ways from existing
functions to functions created on the fly. Usually, inline structure
definitions or lambda functions are used to provide the next-value
functionality.

For example, the following shows how to use a lambda function to satisfy
the ``next_value_func`` parameter for a vital::iterator of type int::

  int a[] = { 0, 1, 2, 3 };
  using iterator_t = vital::iterator< int >;
  iterator_t it( []() ->iterator_t::reference {
    static size_t i = 0;
    if( i == 4 ) { VITAL_THROW( vital::stop_iteration_exception, "container-name" ); }
    return a[i++];
  } );

Similarly an inline structure that overloads operator() can be provided if
more state needs to be tracked::

  using iterator_t = vital::iterator< int >;
  struct next_int_generator
  {
    int *a;
    size_t len;
    size_t idx;

    next_int_generator(int *a, size_t len )
     : a( a )
     , len( len )
     , idx( 0 )
    {}

    iterator_t::reference operator()()
    {
      if( idx == len ) { VITAL_THROW( vital::stop_iteration_exception, "container-name" ); }
      return a[idx++];
    }
  };
  int a[] = {0, 1, 2, 3};
  iterator_t it( next_int_generator(a, 4) );

.. doxygenclass:: kwiver::vital::base_iterator
   :project: kwiver
   :members:

.. doxygenclass:: kwiver::vital::iterator
   :project: kwiver
   :members:

.. doxygenclass:: kwiver::vital::const_iterator
   :project: kwiver
   :members:

References
^^^^^^^^^^
In creating this structure, the following were referenced for what composes
input iterators as well as how this fits into C++ class and function
definitions:

* `How to implement an stl style iterator`_
* `cplusplus.com: Iterator Traits`_
* `cplusplus.com: Input Iterator`_

.. _How to implement an stl style iterator: https://stackoverflow.com/questions/8054273/how-to-implement-an-stl-style-iterator-and-avoid-common-pitfalls
.. _cplusplus.com\: Iterator Traits: http://www.cplusplus.com/reference/iterator/iterator_traits/
.. _cplusplus.com\: Input Iterator: http://www.cplusplus.com/reference/iterator/InputIterator/

.. _vital_iterable:

Iterable Mixin
--------------
This mixin is intended to allow containers implemented in Vital to expose an
iteration interface via the C++ standard ``begin`` and ``end`` methods.

.. doxygenclass:: kwiver::vital::iterable
   :project: kwiver
   :members:
