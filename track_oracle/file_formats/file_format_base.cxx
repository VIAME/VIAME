// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "file_format_base.h"

#include <kwiversys/RegularExpression.hxx>

#include <vul/vul_file.h>
#include <vul/vul_string.h>

#include <track_oracle/core/schema_algorithm.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::istream;
using std::map;
using std::ostream;
using std::string;
using std::vector;

namespace // anon
{

string
glob_to_regexp_string( const string& glob )
{
  //
  // This code is lifted straight from vul_file_iterator.cxx.
  //

  static const string meta_chars = "^$.[()|?+*\\";

  string baseglob = vul_file::basename(glob);
  string::iterator i = baseglob.begin();
  bool prev_slash=false, in_sqr_brackets=false;
  //assemble the Regexp string
  string re = "^"; // match the start of the string
  while (i != baseglob.end())
  {
    const char& c = *i;
    if (c =='\\' && !prev_slash)
    {
      prev_slash = true;
    }
    else if (prev_slash)
    {
      prev_slash = false;
      re.append(1,('\\'));
      re.append(1,*i);
    }
    else if (c=='[' && !in_sqr_brackets)
    {
      in_sqr_brackets = true;
      re.append(1,'[');
    }
    else if (c==']' && in_sqr_brackets)
    {
      in_sqr_brackets = false;
      re.append(1,']');
    }
    else if (c=='?' && !in_sqr_brackets)
    {
      re.append(1,'.');
    }
    else if (c=='*' && !in_sqr_brackets)
    {
      re.append( ".*" );
    }
    else if (meta_chars.find_first_of( c ) != string::npos)
    {
      re.append( "\\" );
      re.append(1, c );
    }
    else
    {
      re.append(1, c);
    }

    ++i;
  }
  // match the end of the string
  re += '$';

  return re;
}

} // anon

namespace kwiver {
namespace track_oracle {

file_format_base
::file_format_base( file_format_enum fmt,
                    const string& desc
  )
  : type( fmt ), description( desc )
{
}

file_format_base
::~file_format_base()
{
}

string
file_format_base
::format_description() const
{
  return this->description;
}

void
file_format_base
::set_format_description( const string& d )
{
  this->description = d;
}

file_format_enum
file_format_base
::get_format() const
{
  return this->type;
}

vector< string >
file_format_base
::format_globs() const
{
  return this->globs;
}

bool
file_format_base
::filename_matches_globs( string fn ) const
{
  vul_string_downcase( fn );
  for (size_t i=0; i<this->globs.size(); ++i)
  {
    kwiversys::RegularExpression r( glob_to_regexp_string( this->globs[i] ));
    if (r.find( fn )) return true;
  }
  return false;
}

map< field_handle_type, track_base_impl::schema_position_type >
file_format_base
::enumerate_schema() const
{
  track_base_impl* t = this->schema_instance();
  map< field_handle_type, track_base_impl::schema_position_type > ret = t->list_schema_elements();
  delete t;
  return ret;
}

file_format_reader_opts_base&
file_format_base
::options()
{
  return this->null_opts;
}

bool
file_format_base
::read( istream&,
        track_handle_list_type& ) const
{
  // by default, not implemented
  LOG_ERROR( main_logger, "Stream reading not supported for " << file_format_type::to_string( this->type ) << " tracks" );
  return false;
}

bool
file_format_base
::write( const string&,
         const track_handle_list_type& ) const
{
  // by default, not implemented
  LOG_ERROR( main_logger, "File writing not supported for " << file_format_type::to_string( this->type ) << " tracks" );
  return false;
}

bool
file_format_base
::write( ostream&,
         const track_handle_list_type& ) const
{
  // by default, not implemented
  LOG_ERROR( main_logger, "Stream writing not supported for " << file_format_type::to_string( this->type ) << " tracks" );
  return false;
}

bool
file_format_base
::read( const string& fn,
        track_handle_list_type& tracks,
        const track_base_impl& required_fields,
        vector< element_descriptor >& missing_fields ) const
{
  if (! this->read( fn, tracks )) return false;
  missing_fields = schema_algorithm::name_missing_fields( required_fields, tracks );
  return true;
}

bool
file_format_base
::read( istream& is,
        track_handle_list_type& tracks,
        const track_base_impl& required_fields,
        vector< element_descriptor >& missing_fields ) const
{
  if (! this->read( is, tracks )) return false;
  missing_fields = schema_algorithm::name_missing_fields( required_fields, tracks );
  return true;
}

} // ...track_oracle
} // ...kwiver
