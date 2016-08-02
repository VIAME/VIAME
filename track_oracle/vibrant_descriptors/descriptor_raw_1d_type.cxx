/*ckwg +5
 * Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "descriptor_raw_1d_type.h"

using std::ostream;
using std::istream;
using std::streamsize;
using std::ios;

namespace kwiver {
namespace track_oracle {

descriptor_raw_1d_type
::descriptor_raw_1d_type( const vnl_vector<double>& d )
{
  this->data.resize( d.size() );
  d.copy_out( &(this->data[0]) );
}

ostream&
operator<<( ostream& os, const descriptor_raw_1d_type& d )
{
  os << "<descriptor type=\"raw\">\n";
  os << "  <vector length=\"" << d.data.size() << "\" value=\"";
  streamsize cur_prec = os.precision();
  os.precision( 15 );
  for (size_t i=0; i<d.data.size(); ++i)
  {
    os << d.data[i] << " ";
  }
  os.precision( cur_prec );
  os << "\"/>\n</descriptor>\n";
  return os;
}

istream&
operator>>( istream& is, descriptor_raw_1d_type& /* d */ )
{
  // TODO
  is.setstate( ios::failbit );
  return is;
}

} // ...track_oracle
} // ...kwiver
