#include <iostream>

#include <track_oracle/element_descriptor.h>
#include <track_oracle/track_base.h>
#include <track_oracle/track_field.h>
#include <track_oracle/track_oracle_core.h>
#include <track_oracle/schema_algorithm.h>
#include <track_oracle/track_field_functor.h>
#include <track_oracle/data_terms/data_terms.h>

//
// Trying to resolve the linking error:
//
// Undefined symbols for architecture x86_64:
//  "std::__1::basic_ostream<char, std::__1::char_traits<char> >&
//     kwiver::track_oracle::operator<< < vgl_box_2d<double> >
//         (  std::__1::basic_ostream<char, std::__1::char_traits<char> >&,
//            kwiver::track_oracle::track_field_io_proxy<vgl_box_2d<double> > const&)",
// referenced from:  _main in test_template.cxx.o
//

using namespace kwiver::track_oracle;

// works
#if 0
template<> std::ostream& kwiver::track_oracle::operator<< <track_field< dt::tracking::bounding_box >::Type> (std::ostream& os, const kwiver::track_oracle::track_field_io_proxy<track_field< dt::tracking::bounding_box >::Type >& foo )
{
  return foo.io_ptr->to_stream( os, foo.val );
}
#endif

// also works
#if 0
template<typename T> std::ostream& kwiver::track_oracle::operator<<(std::ostream& os, const kwiver::track_oracle::track_field_io_proxy< T >& foo )
{
  return foo.io_ptr->to_stream( os, foo.val );
}
#endif

// also works
#if 0
#include <track_oracle/track_field_io_proxy.txx>
template std::ostream& ::kwiver::track_oracle::operator<<(std::ostream& os, const kwiver::track_oracle::track_field_io_proxy< vgl_box_2d<double> > & );
#endif


int main(int, char *[])
{
  track_field< dt::tracking::bounding_box > bb;
  track_field< dt::tracking::bounding_box >::Type bb_t;
  std::cout << bb.io_fmt( bb_t ) << std::endl;

  track_field< int > di("bar");
  std::cout << di() << std::endl;
  track_field< vgl_box_2d<double> > db("foo");
  std::cout << db() << std::endl;

#if 0
  track_field< dt::tracking::external_id > id;
  track_field< dt::tracking::external_id >::Type id_t;

  track_field< dt::utility::state_flags > sf;
  track_field< dt::utility::state_flags >::Type sf_t;

  // works
  std::cout << id.io_fmt( id_t ) << std::endl;
  // also fails with same error!
  std::cout << sf.io_fmt( sf_t ) << std::endl;

#endif
}


// looked at:
//
// http://clang.llvm.org/docs/LTOVisibility.html
//
// http://stackoverflow.com/questions/32830905/clang-c-template-singleton-in-plugins
//
// https://issues.apache.org/jira/browse/PARQUET-659
//
