/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Implementation of the OpenCV FileStorage subset reader and writer

#include "opencv_yaml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace viame {

namespace file_io {

namespace {

// ----------------------------------------------------------------------------
std::string
trim( std::string const& text )
{
  auto const first = text.find_first_not_of( " \t\r\n" );

  if( first == std::string::npos )
  {
    return std::string();
  }

  auto const last = text.find_last_not_of( " \t\r\n" );
  return text.substr( first, last - first + 1 );
}

// ----------------------------------------------------------------------------
/// Whether \p text is an integer literal and nothing else.
bool
is_integer_literal( std::string const& text )
{
  if( text.empty() )
  {
    return false;
  }

  size_t index = ( text[ 0 ] == '-' || text[ 0 ] == '+' ) ? 1 : 0;

  if( index >= text.size() )
  {
    return false;
  }

  for( ; index < text.size(); ++index )
  {
    if( text[ index ] < '0' || text[ index ] > '9' )
    {
      return false;
    }
  }

  return true;
}

// ----------------------------------------------------------------------------
/// Whether \p text is a number literal of any kind.
bool
is_number_literal( std::string const& text )
{
  if( text.empty() )
  {
    return false;
  }

  char* end = nullptr;
  std::strtod( text.c_str(), &end );

  // `0.` and `1.` are numbers here, which strtod already accepts; what this
  // rejects is a trailing anything, so `1.yml` stays a string.
  return end != nullptr && *end == '\0';
}

// ----------------------------------------------------------------------------
/// One scalar, as the node its spelling makes it.
node
scalar_node( std::string const& raw )
{
  auto text = trim( raw );

  if( text.size() >= 2 &&
      ( ( text.front() == '"' && text.back() == '"' ) ||
        ( text.front() == '\'' && text.back() == '\'' ) ) )
  {
    // Quoted: a string whatever it looks like, which is how OpenCV keeps a
    // numeric-looking name a name.
    return node( text.substr( 1, text.size() - 2 ) );
  }

  if( text.empty() || text == "~" || text == "null" )
  {
    return node();
  }

  if( is_integer_literal( text ) )
  {
    return node( static_cast< long long >( std::strtoll( text.c_str(),
                                                         nullptr, 10 ) ) );
  }

  if( is_number_literal( text ) )
  {
    return node( std::strtod( text.c_str(), nullptr ) );
  }

  return node( text );
}

// ----------------------------------------------------------------------------
/// Split a flow sequence's body -- what is between `[` and `]` -- on commas.
std::vector< std::string >
split_flow( std::string const& body )
{
  std::vector< std::string > out;
  std::string current;
  int depth = 0;
  bool quoted = false;
  char quote = '\0';

  for( char c : body )
  {
    if( quoted )
    {
      current.push_back( c );
      if( c == quote ) { quoted = false; }
      continue;
    }

    if( c == '"' || c == '\'' )
    {
      quoted = true;
      quote = c;
      current.push_back( c );
      continue;
    }

    if( c == '[' || c == '{' ) { ++depth; }
    if( c == ']' || c == '}' ) { --depth; }

    if( c == ',' && depth == 0 )
    {
      out.push_back( current );
      current.clear();
      continue;
    }

    current.push_back( c );
  }

  auto const last = trim( current );
  if( !last.empty() )
  {
    out.push_back( current );
  }

  return out;
}

// ----------------------------------------------------------------------------
/// A line of a YAML document, with its indentation measured.
struct line
{
  size_t indent = 0;
  std::string text;
};

// ----------------------------------------------------------------------------
/// The lines of \p text, with comments and blanks dropped.
std::vector< line >
lines_of( std::string const& text )
{
  std::vector< line > out;
  std::istringstream stream( text );
  std::string raw;

  while( std::getline( stream, raw ) )
  {
    if( !raw.empty() && raw.back() == '\r' )
    {
      raw.pop_back();
    }

    size_t indent = 0;
    while( indent < raw.size() && raw[ indent ] == ' ' )
    {
      ++indent;
    }

    auto body = raw.substr( indent );

    // A comment is only a comment at the start of a line: `#` appears inside
    // a quoted string in some documents and must not end the value there.
    if( body.empty() || body[ 0 ] == '#' )
    {
      continue;
    }

    if( body.rfind( "%YAML", 0 ) == 0 || body == "---" || body == "..." )
    {
      continue;
    }

    out.push_back( line{ indent, body } );
  }

  return out;
}

// ----------------------------------------------------------------------------
/// Where a scalar's value ends and its `!!tag` begins.
struct tagged
{
  std::string tag;
  std::string value;
};

tagged
split_tag( std::string const& text )
{
  auto value = trim( text );

  if( value.rfind( "!!", 0 ) != 0 )
  {
    return tagged{ std::string(), value };
  }

  auto const space = value.find_first_of( " \t" );

  if( space == std::string::npos )
  {
    return tagged{ value, std::string() };
  }

  return tagged{ value.substr( 0, space ), trim( value.substr( space ) ) };
}

// ----------------------------------------------------------------------------
class yaml_parser
{
public:
  explicit yaml_parser( std::vector< line > lines )
    : lines_( std::move( lines ) )
  {}

  node parse_document()
  {
    if( lines_.empty() )
    {
      return node::map_of( {} );
    }

    return parse_map( lines_[ 0 ].indent, false );
  }

private:
  std::vector< line > lines_;
  size_t at_ = 0;

  bool done() const { return at_ >= lines_.size(); }
  line const& current() const { return lines_[ at_ ]; }

  // --------------------------------------------------------------------
  /// A block map: every line at \p indent that is `key: ...`.
  node parse_map( size_t indent, bool matrix )
  {
    std::vector< node::entry > entries;

    while( !done() && current().indent == indent )
    {
      auto const body = current().text;

      if( body.rfind( "- ", 0 ) == 0 || body == "-" )
      {
        break;
      }

      auto const colon = find_key_end( body );

      if( colon == std::string::npos )
      {
        throw parse_error( "not a key: '" + body + "'" );
      }

      auto const key = unquote( trim( body.substr( 0, colon ) ) );
      auto const rest = trim( body.substr( colon + 1 ) );

      ++at_;
      entries.emplace_back( key, parse_value( rest, indent ) );
    }

    return node::map_of( std::move( entries ), matrix );
  }

  // --------------------------------------------------------------------
  /// A block sequence: every line at \p indent starting with `- `.
  node parse_sequence( size_t indent )
  {
    std::vector< node > values;

    while( !done() && current().indent == indent &&
           ( current().text.rfind( "- ", 0 ) == 0 || current().text == "-" ) )
    {
      auto const body = current().text.size() > 1
                        ? trim( current().text.substr( 2 ) )
                        : std::string();

      // `- key: value` opens a map whose first key sits on the dash's line,
      // so the map's indentation is where that key starts.
      auto const inline_colon = find_key_end( body );

      if( inline_colon != std::string::npos && !body.empty() &&
          body[ 0 ] != '[' && body[ 0 ] != '{' )
      {
        auto const key = unquote( trim( body.substr( 0, inline_colon ) ) );
        auto const rest = trim( body.substr( inline_colon + 1 ) );
        auto const key_indent = indent + 2;

        ++at_;
        std::vector< node::entry > entries;
        entries.emplace_back( key, parse_value( rest, key_indent ) );

        while( !done() && current().indent == key_indent &&
               current().text.rfind( "- ", 0 ) != 0 )
        {
          auto const more = current().text;
          auto const colon = find_key_end( more );

          if( colon == std::string::npos )
          {
            break;
          }

          auto const next_key = unquote( trim( more.substr( 0, colon ) ) );
          auto const next_rest = trim( more.substr( colon + 1 ) );

          ++at_;
          entries.emplace_back( next_key,
                                parse_value( next_rest, key_indent ) );
        }

        values.push_back( node::map_of( std::move( entries ) ) );
        continue;
      }

      ++at_;
      values.push_back( parse_value( body, indent ) );
    }

    return node::sequence_of( std::move( values ) );
  }

  // --------------------------------------------------------------------
  /// The value that follows a `key:`, which may be on this line or below.
  node parse_value( std::string const& rest, size_t parent_indent )
  {
    auto const split = split_tag( rest );
    auto const is_matrix = ( split.tag == "!!opencv-matrix" );
    auto const value = split.value;

    if( !value.empty() && value[ 0 ] == '[' )
    {
      return parse_flow_sequence( value );
    }

    if( !value.empty() && value[ 0 ] == '{' )
    {
      return parse_flow_map( value );
    }

    if( !value.empty() )
    {
      return scalar_node( value );
    }

    // Nothing after the colon: whatever is indented under it.
    if( done() || current().indent <= parent_indent )
    {
      return is_matrix ? node::map_of( {}, true ) : node();
    }

    auto const child = current().indent;

    if( current().text.rfind( "- ", 0 ) == 0 || current().text == "-" )
    {
      return parse_sequence( child );
    }

    return parse_map( child, is_matrix );
  }

  // --------------------------------------------------------------------
  /// A flow sequence, which may run over several lines.
  node parse_flow_sequence( std::string const& opening )
  {
    auto const body = gather_brackets( opening, '[', ']' );

    std::vector< node > values;
    for( auto const& piece : split_flow( body ) )
    {
      auto const text = trim( piece );

      if( !text.empty() && text[ 0 ] == '[' )
      {
        values.push_back( parse_flow_sequence( text ) );
      }
      else if( !text.empty() && text[ 0 ] == '{' )
      {
        values.push_back( parse_flow_map( text ) );
      }
      else
      {
        values.push_back( scalar_node( text ) );
      }
    }

    return node::sequence_of( std::move( values ) );
  }

  // --------------------------------------------------------------------
  node parse_flow_map( std::string const& opening )
  {
    auto const body = gather_brackets( opening, '{', '}' );

    std::vector< node::entry > entries;
    for( auto const& piece : split_flow( body ) )
    {
      auto const text = trim( piece );
      auto const colon = find_key_end( text );

      if( colon == std::string::npos )
      {
        continue;
      }

      entries.emplace_back( unquote( trim( text.substr( 0, colon ) ) ),
                            scalar_node( text.substr( colon + 1 ) ) );
    }

    return node::map_of( std::move( entries ) );
  }

  // --------------------------------------------------------------------
  /// Everything between \p open and its matching \p close, reading on until
  /// the brackets balance -- a matrix's data runs over as many lines as it
  /// needs, and this is what puts it back together.
  std::string gather_brackets( std::string const& opening, char open,
                               char close )
  {
    std::string text = opening;
    int depth = balance( text, open, close );

    while( depth > 0 && !done() )
    {
      text += " " + current().text;
      depth = balance( text, open, close );
      ++at_;
    }

    auto const first = text.find( open );
    auto const last = text.rfind( close );

    if( first == std::string::npos || last == std::string::npos ||
        last < first )
    {
      throw parse_error( "unterminated '" + std::string( 1, open ) + "'" );
    }

    return text.substr( first + 1, last - first - 1 );
  }

  static int balance( std::string const& text, char open, char close )
  {
    int depth = 0;
    bool quoted = false;
    char quote = '\0';

    for( char c : text )
    {
      if( quoted )
      {
        if( c == quote ) { quoted = false; }
        continue;
      }

      if( c == '"' || c == '\'' ) { quoted = true; quote = c; continue; }
      if( c == open ) { ++depth; }
      if( c == close ) { --depth; }
    }

    return depth;
  }

  // --------------------------------------------------------------------
  /// Where a `key:` ends, skipping a colon inside quotes.
  static size_t find_key_end( std::string const& text )
  {
    bool quoted = false;
    char quote = '\0';

    for( size_t index = 0; index < text.size(); ++index )
    {
      char const c = text[ index ];

      if( quoted )
      {
        if( c == quote ) { quoted = false; }
        continue;
      }

      if( c == '"' || c == '\'' ) { quoted = true; quote = c; continue; }

      if( c == ':' )
      {
        // A colon ends a key only when a space or the end of the line
        // follows it, which is what keeps `12:30:00` one scalar.
        if( index + 1 == text.size() || text[ index + 1 ] == ' ' )
        {
          return index;
        }
      }
    }

    return std::string::npos;
  }

  static std::string unquote( std::string const& text )
  {
    if( text.size() >= 2 &&
        ( ( text.front() == '"' && text.back() == '"' ) ||
          ( text.front() == '\'' && text.back() == '\'' ) ) )
    {
      return text.substr( 1, text.size() - 2 );
    }

    return text;
  }
};

// ----------------------------------------------------------------------------
// XML
// ----------------------------------------------------------------------------

class xml_parser
{
public:
  explicit xml_parser( std::string const& text ) : text_( text ) {}

  node parse_document()
  {
    skip_prologue();

    auto const root = read_element();

    if( root.first != "opencv_storage" )
    {
      throw parse_error( "XML root is <" + root.first +
                         ">, expected <opencv_storage>" );
    }

    return root.second;
  }

private:
  std::string const& text_;
  size_t at_ = 0;

  void skip_prologue()
  {
    while( true )
    {
      skip_space();

      if( at_ + 1 >= text_.size() || text_[ at_ ] != '<' )
      {
        return;
      }

      if( text_[ at_ + 1 ] == '?' || text_[ at_ + 1 ] == '!' )
      {
        auto const end = text_.find( '>', at_ );
        if( end == std::string::npos )
        {
          throw parse_error( "unterminated XML prologue" );
        }
        at_ = end + 1;
        continue;
      }

      return;
    }
  }

  void skip_space()
  {
    while( at_ < text_.size() && std::isspace(
             static_cast< unsigned char >( text_[ at_ ] ) ) )
    {
      ++at_;
    }
  }

  /// One element and its children, as (name, node).
  std::pair< std::string, node > read_element()
  {
    skip_space();

    if( at_ >= text_.size() || text_[ at_ ] != '<' )
    {
      throw parse_error( "expected an XML element" );
    }

    auto const close = text_.find( '>', at_ );

    if( close == std::string::npos )
    {
      throw parse_error( "unterminated XML tag" );
    }

    auto tag = text_.substr( at_ + 1, close - at_ - 1 );
    at_ = close + 1;

    bool const empty = !tag.empty() && tag.back() == '/';
    if( empty )
    {
      tag.pop_back();
    }

    auto const space = tag.find_first_of( " \t\r\n" );
    auto const name = trim( space == std::string::npos ? tag
                                                       : tag.substr( 0, space ) );

    if( empty )
    {
      return { name, node() };
    }

    // Children, text, or both. A document VIAME writes has one or the other.
    std::string text;
    std::vector< node::entry > entries;
    std::vector< node > values;
    bool sequence = false;

    while( true )
    {
      auto const next = text_.find( '<', at_ );

      if( next == std::string::npos )
      {
        throw parse_error( "unterminated <" + name + ">" );
      }

      text += text_.substr( at_, next - at_ );
      at_ = next;

      if( at_ + 1 < text_.size() && text_[ at_ + 1 ] == '/' )
      {
        auto const end = text_.find( '>', at_ );
        if( end == std::string::npos )
        {
          throw parse_error( "unterminated </" + name + ">" );
        }
        at_ = end + 1;
        break;
      }

      auto const child = read_element();

      // `<_>` is FileStorage's sequence entry; a document mixing it with
      // named children is one this reader has never seen and should not
      // guess at.
      if( child.first == "_" )
      {
        sequence = true;
        values.push_back( child.second );
      }
      else
      {
        entries.push_back( child );
      }
    }

    if( sequence )
    {
      return { name, node::sequence_of( std::move( values ) ) };
    }

    if( !entries.empty() )
    {
      return { name, node::map_of( std::move( entries ) ) };
    }

    return { name, scalar_node( unescape( text ) ) };
  }

  static std::string unescape( std::string const& text )
  {
    static std::pair< char const*, char > const table[] = {
      { "&lt;", '<' }, { "&gt;", '>' }, { "&quot;", '"' },
      { "&apos;", '\'' }, { "&amp;", '&' },
    };

    std::string out;
    for( size_t index = 0; index < text.size(); )
    {
      bool replaced = false;

      if( text[ index ] == '&' )
      {
        for( auto const& pair : table )
        {
          auto const length = std::string( pair.first ).size();
          if( text.compare( index, length, pair.first ) == 0 )
          {
            out.push_back( pair.second );
            index += length;
            replaced = true;
            break;
          }
        }
      }

      if( !replaced )
      {
        out.push_back( text[ index ] );
        ++index;
      }
    }

    return out;
  }
};

// ----------------------------------------------------------------------------
// Writing
// ----------------------------------------------------------------------------

// Where OpenCV's YAML emitter breaks a line. Measured from what it produces
// rather than read out of its source: it wraps before a token that would end
// past column 72, and a continuation line is seven spaces and then the token
// with no space of its own. Four sequences of differently sized numbers pin
// the margin to exactly 72 -- a token ending at 72 stays, one ending at 73
// wraps.
size_t const WRAP_MARGIN = 72;
char const* const CONTINUATION = "       ";

// ----------------------------------------------------------------------------
/// Accumulate tokens into a flow sequence, wrapping as OpenCV does.
class flow_writer
{
public:
  flow_writer( std::ostream& out, std::string const& opening )
    : out_( out ), column_( opening.size() )
  {
    out_ << opening;
  }

  void add( std::string const& token )
  {
    if( first_ )
    {
      out_ << " " << token;
      column_ += 1 + token.size();
      first_ = false;
      return;
    }

    out_ << ",";
    ++column_;

    if( column_ + 1 + token.size() > WRAP_MARGIN )
    {
      out_ << "\n" << CONTINUATION << token;
      column_ = std::string( CONTINUATION ).size() + token.size();
      return;
    }

    out_ << " " << token;
    column_ += 1 + token.size();
  }

  void close()
  {
    out_ << " ]";
  }

private:
  std::ostream& out_;
  size_t column_;
  bool first_ = true;
};

// ----------------------------------------------------------------------------
std::string
scalar_to_string( node const& value )
{
  switch( value.type() )
  {
    case node::kind::integer:
      return std::to_string( value.as_integer() );

    case node::kind::real:
      return double_to_string( value.as_double() );

    case node::kind::string:
    {
      auto const& text = value.as_string();

      // OpenCV quotes a string only when leaving it bare would reparse as
      // something else: empty, numeric looking, or carrying a character the
      // block scalar rules would take.
      if( text.empty() || is_number_literal( text ) ||
          text.find_first_of( ":#[]{}\",'" ) != std::string::npos ||
          text.front() == ' ' || text.back() == ' ' )
      {
        return "\"" + text + "\"";
      }

      return text;
    }

    case node::kind::none:
      return std::string();

    default:
      throw parse_error( "not a scalar" );
  }
}

void write_node( std::ostream& out, node const& value, size_t indent );

// ----------------------------------------------------------------------------
void
write_matrix( std::ostream& out, node const& value, size_t indent )
{
  std::string const pad( indent, ' ' );
  auto const dt = value.matrix_type();

  out << " !!opencv-matrix\n";
  out << pad << "rows: " << value.matrix_rows() << "\n";
  out << pad << "cols: " << value.matrix_cols() << "\n";
  out << pad << "dt: " << dt << "\n";

  flow_writer flow( out, pad + "data: [" );

  for( double element : value.matrix_data() )
  {
    if( dt == "d" )
    {
      flow.add( double_to_string( element ) );
    }
    else if( dt == "f" )
    {
      flow.add( float_to_string( static_cast< float >( element ) ) );
    }
    else
    {
      flow.add( std::to_string( static_cast< long long >( element ) ) );
    }
  }

  flow.close();
  out << "\n";
}

// ----------------------------------------------------------------------------
void
write_map_body( std::ostream& out, node const& value, size_t indent )
{
  std::string const pad( indent, ' ' );

  for( auto const& entry : value.entries() )
  {
    out << pad << entry.first << ":";
    write_node( out, entry.second, indent + 3 );
  }
}

// ----------------------------------------------------------------------------
void
write_node( std::ostream& out, node const& value, size_t indent )
{
  std::string const pad( indent, ' ' );

  if( value.is_matrix() )
  {
    write_matrix( out, value, indent );
    return;
  }

  if( value.is_map() )
  {
    out << "\n";
    write_map_body( out, value, indent );
    return;
  }

  if( value.is_sequence() )
  {
    // Scalars go in the flow form, which is what a matrix's data is and what
    // every calibration file holds. Anything else goes in the block form,
    // one `- ` per element -- that is what an XML document's `<_>` entries
    // become when they are written back out as YAML.
    bool const flat = std::all_of(
      value.values().begin(), value.values().end(),
      []( node const& element )
      {
        return !element.is_map() && !element.is_sequence();
      } );

    if( flat )
    {
      flow_writer flow( out, " [" );

      for( auto const& element : value.values() )
      {
        flow.add( scalar_to_string( element ) );
      }

      flow.close();
      out << "\n";
      return;
    }

    out << "\n";

    for( auto const& element : value.values() )
    {
      out << pad << "-";

      if( element.is_map() && !element.is_matrix() )
      {
        // The first key sits on the dash's line, and the rest line up under
        // it -- two columns past the dash, which is where `- ` ends.
        auto const& entries = element.entries();

        for( size_t index = 0; index < entries.size(); ++index )
        {
          if( index > 0 )
          {
            out << std::string( indent + 2, ' ' );
          }
          else
          {
            out << " ";
          }

          out << entries[ index ].first << ":";
          write_node( out, entries[ index ].second, indent + 5 );
        }

        if( entries.empty() )
        {
          out << "\n";
        }

        continue;
      }

      write_node( out, element, indent + 2 );
    }

    return;
  }

  out << " " << scalar_to_string( value ) << "\n";
}

} // namespace

// ----------------------------------------------------------------------------
// node
// ----------------------------------------------------------------------------

node
::node()
  : kind_( kind::none ), matrix_( false ), integer_( 0 ), real_( 0.0 )
{}

node
::node( long long value )
  : kind_( kind::integer ), matrix_( false ), integer_( value ), real_( 0.0 )
{}

node
::node( double value )
  : kind_( kind::real ), matrix_( false ), integer_( 0 ), real_( value )
{}

node
::node( std::string value )
  : kind_( kind::string ), matrix_( false ), integer_( 0 ), real_( 0.0 ),
    string_( std::move( value ) )
{}

// ----------------------------------------------------------------------------
node
node
::sequence_of( std::vector< node > values )
{
  node out;
  out.kind_ = kind::sequence;
  out.values_ = std::move( values );
  return out;
}

// ----------------------------------------------------------------------------
node
node
::map_of( std::vector< entry > entries, bool matrix )
{
  node out;
  out.kind_ = kind::map;
  out.matrix_ = matrix;
  out.entries_ = std::move( entries );
  return out;
}

// ----------------------------------------------------------------------------
node
node
::matrix( int rows, int cols, std::string dt,
          std::vector< double > const& data )
{
  if( static_cast< size_t >( rows ) * static_cast< size_t >( cols ) !=
      data.size() )
  {
    throw parse_error( "matrix is " + std::to_string( rows ) + "x" +
                       std::to_string( cols ) + " but has " +
                       std::to_string( data.size() ) + " values" );
  }

  std::vector< node > values;
  values.reserve( data.size() );
  for( double element : data )
  {
    values.push_back( node( element ) );
  }

  std::vector< entry > entries;
  entries.emplace_back( "rows", node( static_cast< long long >( rows ) ) );
  entries.emplace_back( "cols", node( static_cast< long long >( cols ) ) );
  entries.emplace_back( "dt", node( std::move( dt ) ) );
  entries.emplace_back( "data", sequence_of( std::move( values ) ) );

  return map_of( std::move( entries ), true );
}

// ----------------------------------------------------------------------------
long long
node
::as_integer() const
{
  if( kind_ == kind::integer ) { return integer_; }
  if( kind_ == kind::real ) { return static_cast< long long >( real_ ); }

  throw parse_error( "node is not a number" );
}

// ----------------------------------------------------------------------------
double
node
::as_double() const
{
  if( kind_ == kind::real ) { return real_; }
  if( kind_ == kind::integer ) { return static_cast< double >( integer_ ); }

  throw parse_error( "node is not a number" );
}

// ----------------------------------------------------------------------------
std::string const&
node
::as_string() const
{
  if( kind_ != kind::string )
  {
    throw parse_error( "node is not a string" );
  }

  return string_;
}

// ----------------------------------------------------------------------------
node const&
node
::operator[]( std::string const& key ) const
{
  static node const none;

  for( auto const& entry : entries_ )
  {
    if( entry.first == key )
    {
      return entry.second;
    }
  }

  return none;
}

// ----------------------------------------------------------------------------
bool
node
::has( std::string const& key ) const
{
  for( auto const& entry : entries_ )
  {
    if( entry.first == key )
    {
      return true;
    }
  }

  return false;
}

// ----------------------------------------------------------------------------
int
node
::matrix_rows() const
{
  return static_cast< int >( ( *this )[ "rows" ].as_integer() );
}

int
node
::matrix_cols() const
{
  return static_cast< int >( ( *this )[ "cols" ].as_integer() );
}

std::string
node
::matrix_type() const
{
  auto const& dt = ( *this )[ "dt" ];
  return dt.is_string() ? dt.as_string() : std::string( "d" );
}

// ----------------------------------------------------------------------------
std::vector< double >
node
::matrix_data() const
{
  if( !is_matrix() )
  {
    throw parse_error( "node is not an opencv-matrix" );
  }

  auto const& data = ( *this )[ "data" ];

  if( !data.is_sequence() )
  {
    throw parse_error( "matrix has no data sequence" );
  }

  std::vector< double > out;
  out.reserve( data.values().size() );

  for( auto const& element : data.values() )
  {
    out.push_back( element.as_double() );
  }

  auto const expected = static_cast< size_t >( matrix_rows() ) *
                        static_cast< size_t >( matrix_cols() );

  if( out.size() != expected )
  {
    throw parse_error( "matrix says " + std::to_string( expected ) +
                       " values but has " + std::to_string( out.size() ) );
  }

  return out;
}

// ----------------------------------------------------------------------------
// Number formatting
// ----------------------------------------------------------------------------

std::string
double_to_string( double value )
{
  if( std::isnan( value ) ) { return ".NaN"; }
  if( std::isinf( value ) ) { return value > 0 ? ".Inf" : "-.Inf"; }

  auto const rounded = std::llround( value );

  if( static_cast< double >( rounded ) == value )
  {
    return std::to_string( rounded ) + ".";
  }

  char buffer[ 64 ];
  std::snprintf( buffer, sizeof( buffer ), "%.16e", value );
  return buffer;
}

// ----------------------------------------------------------------------------
std::string
float_to_string( float value )
{
  if( std::isnan( value ) ) { return ".NaN"; }
  if( std::isinf( value ) ) { return value > 0 ? ".Inf" : "-.Inf"; }

  auto const rounded = std::llround( value );

  if( static_cast< float >( rounded ) == value )
  {
    return std::to_string( rounded ) + ".";
  }

  char buffer[ 64 ];
  std::snprintf( buffer, sizeof( buffer ), "%.8e",
                 static_cast< double >( value ) );
  return buffer;
}

// ----------------------------------------------------------------------------
// Entry points
// ----------------------------------------------------------------------------

node
parse_yaml( std::string const& text )
{
  yaml_parser parser( lines_of( text ) );
  return parser.parse_document();
}

// ----------------------------------------------------------------------------
node
parse_xml( std::string const& text )
{
  xml_parser parser( text );
  return parser.parse_document();
}

// ----------------------------------------------------------------------------
node
read( std::string const& path )
{
  std::ifstream file( path, std::ios::binary );

  if( !file )
  {
    throw parse_error( "could not open " + path );
  }

  std::ostringstream buffer;
  buffer << file.rdbuf();
  auto const text = buffer.str();

  // By content, not by extension: VIAME has `.yml` files and `.xml` files
  // and both have been handed to the wrong reader before.
  auto const first = text.find_first_not_of( " \t\r\n" );

  if( first != std::string::npos && text[ first ] == '<' )
  {
    return parse_xml( text );
  }

  return parse_yaml( text );
}

// ----------------------------------------------------------------------------
std::string
to_yaml( node const& root )
{
  if( !root.is_map() )
  {
    throw parse_error( "the root of a document must be a map" );
  }

  std::ostringstream out;
  out << "%YAML:1.0\n---\n";
  write_map_body( out, root, 0 );

  return out.str();
}

// ----------------------------------------------------------------------------
void
write( std::string const& path, node const& root )
{
  auto const text = to_yaml( root );

  std::ofstream file( path, std::ios::binary );

  if( !file )
  {
    throw parse_error( "could not open " + path + " for writing" );
  }

  file << text;

  if( !file )
  {
    throw parse_error( "could not write " + path );
  }
}

} // namespace file_io

} // namespace viame
