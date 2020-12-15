// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "bounding_box.h"

namespace kwiver {
namespace vital {

//
// Instantiate operators
//
#define instantiate(T)                                                  \
template bounding_box<T> & translate( bounding_box<T>& bbox,            \
                                      bounding_box<T>::vector_type const& pt ); \
template bounding_box<T> scale( bounding_box<T> const& bbox,            \
                                double scale_factor );                  \
template bounding_box<T> intersection( bounding_box<T> const& one,      \
                                       bounding_box<T> const& other )

instantiate( int );
instantiate( double );

#undef instantiate

} }                             // end namespace
