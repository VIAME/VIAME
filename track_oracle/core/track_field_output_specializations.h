// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_FIELD_OUTPUT_SPECIALIZATIONS_H
#define INCL_TRACK_FIELD_OUTPUT_SPECIALIZATIONS_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <set>
#include <utility>
#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/track_field.h>

namespace kwiver {
namespace track_oracle {

// specialization for e.g. frame lists
template< >
TRACK_ORACLE_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< frame_handle_list_type >& f );

template< >
TRACK_ORACLE_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< track_handle_list_type >& f );

template< >
TRACK_ORACLE_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< std::vector< unsigned int> >& f );

template< >
TRACK_ORACLE_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< std::pair<unsigned int, unsigned int> >& f );

template< >
TRACK_ORACLE_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< std::vector< double> >& f );

template< >
TRACK_ORACLE_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< std::vector< std::vector<double> > >& f);

template< >
TRACK_ORACLE_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< std::vector< std::string> >& f );

template< >
TRACK_ORACLE_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< std::set< std::string> >& f );

template< >
TRACK_ORACLE_EXPORT std::ostream&
operator<<( std::ostream& os,
            const track_field< std::map< std::string, double > >& f );

} // ...track_oracle
} // ...kwiver

#endif
