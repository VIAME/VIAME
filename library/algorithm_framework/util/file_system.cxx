// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief File system operations, over `std::filesystem`.
///
/// The semantics are kwiversys's, because two hundred call sites were
/// written against them, and four of them are not `std::filesystem`'s. Each
/// is marked below and recorded in
/// `tests/library/algorithm_framework/test_file_system.cxx`; the recording
/// was taken against kwiversys before this file replaced it, so it is the
/// old behaviour rather than an opinion about the new.
///
/// Nothing here throws. A question about a path that does not exist, or that
/// the caller cannot read, is answered `false` or empty -- which is what the
/// callers expect, and several of them ask precisely because they do not
/// know.

#include <viame/algorithm_framework/util/file_system.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <system_error>

namespace kwiver {

namespace vital {

namespace fs = std::filesystem;

namespace {

constexpr char separator = '/';

// ----------------------------------------------------------------------------
/// Runs of separators collapsed, and no trailing one.
///
/// The root keeps its separator: `"/"` normalises to itself, not to empty.
std::string
normalised( std::string const& path )
{
  std::string out;
  out.reserve( path.size() );

  for( char const c : path )
  {
    if( c == separator && !out.empty() && out.back() == separator )
    {
      continue;
    }

    out.push_back( c );
  }

  if( out.size() > 1 && out.back() == separator )
  {
    out.pop_back();
  }

  return out;
}

} // namespace

// ----------------------------------------------------------------------------
bool
file_exists( std::string const& path )
{
  if( path.empty() )
  {
    return false;
  }

  std::error_code error;
  return fs::exists( path, error ) && !error;
}

// ----------------------------------------------------------------------------
bool
file_is_directory( std::string const& path )
{
  if( path.empty() )
  {
    return false;
  }

  std::error_code error;
  return fs::is_directory( path, error ) && !error;
}

// ----------------------------------------------------------------------------
bool
file_is_full_path( std::string const& path )
{
  return fs::path( path ).is_absolute();
}

// ----------------------------------------------------------------------------
std::string
filename_path( std::string const& path )
{
  // **Not `parent_path`.** The trailing separator goes first, so that
  // `"/a/b/"` is the parent of `b` rather than of the empty name after it.
  auto const whole = normalised( path );
  auto const cut = whole.rfind( separator );

  if( cut == std::string::npos )
  {
    return {};
  }

  if( cut == 0 )
  {
    return std::string( 1, separator );
  }

  return whole.substr( 0, cut );
}

// ----------------------------------------------------------------------------
std::string
filename_name( std::string const& path )
{
  // Deliberately *not* normalised: a path ending in a separator has no last
  // component, and says so.
  auto const cut = path.rfind( separator );

  if( cut == std::string::npos )
  {
    return path;
  }

  return path.substr( cut + 1 );
}

// ----------------------------------------------------------------------------
std::string
filename_last_extension( std::string const& path )
{
  auto const name = filename_name( path );
  auto const dot = name.rfind( '.' );

  if( dot == std::string::npos )
  {
    return {};
  }

  // **Not `path::extension`.** A name that is nothing but an extension is
  // all extension; C++17 says such a name has none.
  return name.substr( dot );
}

// ----------------------------------------------------------------------------
std::string
filename_without_last_extension( std::string const& path )
{
  auto const name = filename_name( path );
  auto const dot = name.rfind( '.' );

  if( dot == std::string::npos )
  {
    return name;
  }

  return name.substr( 0, dot );
}

// ----------------------------------------------------------------------------
std::string
parent_directory( std::string const& path )
{
  return filename_path( path );
}

// ----------------------------------------------------------------------------
void
split_path( std::string const& path, std::vector< std::string >& components )
{
  auto const whole = normalised( path );
  auto const absolute = !whole.empty() && whole.front() == separator;

  // **Not the `std::filesystem::path` iterator.** The first element is the
  // root, and a relative path gets an empty one rather than none, so that
  // the two shapes have the same length and `join_path` can tell them apart.
  components.push_back( absolute ? std::string( 1, separator ) : std::string{} );

  std::string::size_type at = absolute ? 1u : 0u;

  while( at < whole.size() )
  {
    auto const next = whole.find( separator, at );

    if( next == std::string::npos )
    {
      components.push_back( whole.substr( at ) );
      break;
    }

    components.push_back( whole.substr( at, next - at ) );
    at = next + 1;
  }
}

// ----------------------------------------------------------------------------
std::string
join_path( std::vector< std::string > const& components )
{
  if( components.empty() )
  {
    return {};
  }

  std::string out = components.front();

  for( size_t i = 1; i < components.size(); ++i )
  {
    if( !out.empty() && out.back() != separator )
    {
      out.push_back( separator );
    }

    out += components[ i ];
  }

  return out;
}

// ----------------------------------------------------------------------------
std::string
collapse_full_path( std::string const& path, std::string const& base )
{
  fs::path whole( path );

  if( whole.is_relative() )
  {
    whole = fs::path( base ) / whole;
  }

  // `lexically_normal`, not `canonical`: this answers a question about the
  // shape of a path, and the path is often one that has not been written
  // yet.
  return normalised( whole.lexically_normal().string() );
}

// ----------------------------------------------------------------------------
std::string
collapse_full_path( std::string const& path )
{
  return collapse_full_path( path, current_working_directory() );
}

// ----------------------------------------------------------------------------
std::string
real_path( std::string const& path )
{
  std::error_code error;
  auto const resolved = fs::weakly_canonical( path, error );

  if( error )
  {
    return collapse_full_path( path );
  }

  return normalised( resolved.string() );
}

// ----------------------------------------------------------------------------
std::string
current_working_directory()
{
  std::error_code error;
  auto const here = fs::current_path( error );

  if( error )
  {
    return {};
  }

  return here.string();
}

// ----------------------------------------------------------------------------
void
convert_to_unix_slashes( std::string& path )
{
  std::replace( path.begin(), path.end(), '\\', separator );
  path = normalised( path );
}

// ----------------------------------------------------------------------------
bool
make_directory( std::string const& path )
{
  if( path.empty() )
  {
    return false;
  }

  std::error_code error;
  fs::create_directories( path, error );

  // Not the return value: `create_directories` says false when there was
  // nothing to create, and a caller asking for a directory it already has
  // wants yes.
  return file_is_directory( path );
}

// ----------------------------------------------------------------------------
bool
remove_directory( std::string const& path )
{
  if( path.empty() )
  {
    return false;
  }

  std::error_code error;
  fs::remove_all( path, error );

  return !file_exists( path );
}

// ----------------------------------------------------------------------------
bool
remove_file( std::string const& path )
{
  std::error_code error;
  return fs::remove( path, error ) && !error;
}

// ----------------------------------------------------------------------------
std::string
find_file(
  std::string const& name, std::vector< std::string > const& directories )
{
  for( auto const& directory : directories )
  {
    auto const candidate = collapse_full_path( name, directory );

    if( file_exists( candidate ) && !file_is_directory( candidate ) )
    {
      return candidate;
    }
  }

  return {};
}

// ----------------------------------------------------------------------------
std::string
find_program(
  std::string const& name, std::vector< std::string > const& directories )
{
  auto places = directories;

  if( char const* const path = std::getenv( "PATH" ) )
  {
    std::string const whole( path );
    std::string::size_type at = 0;

    while( at <= whole.size() )
    {
      auto const next = whole.find( ':', at );
      auto const stop = ( next == std::string::npos ) ? whole.size() : next;

      if( stop > at )
      {
        places.push_back( whole.substr( at, stop - at ) );
      }

      if( next == std::string::npos )
      {
        break;
      }

      at = next + 1;
    }
  }

  for( auto const& directory : places )
  {
    auto const candidate = collapse_full_path( name, directory );

    std::error_code error;
    auto const status = fs::status( candidate, error );

    if( error || !fs::is_regular_file( status ) )
    {
      continue;
    }

    if( ( status.permissions() & ( fs::perms::owner_exec |
                                   fs::perms::group_exec |
                                   fs::perms::others_exec ) ) !=
        fs::perms::none )
    {
      return candidate;
    }
  }

  return {};
}

// ----------------------------------------------------------------------------
std::vector< std::string >
directory_entries( std::string const& path )
{
  std::vector< std::string > names;

  if( !file_is_directory( path ) )
  {
    return names;
  }

  std::error_code error;
  fs::directory_iterator entries( path, error );

  if( error )
  {
    return names;
  }

  // **Not what `directory_iterator` gives.** `.` and `..` are in the list,
  // because that is what the callers have always been filtering out.
  names.emplace_back( "." );
  names.emplace_back( ".." );

  for( auto const& entry : entries )
  {
    names.emplace_back( entry.path().filename().string() );
  }

  return names;
}

// ----------------------------------------------------------------------------
bool
get_env( std::string const& name, std::string& value )
{
  char const* const found = std::getenv( name.c_str() );

  if( !found )
  {
    return false;
  }

  value = found;
  return true;
}

// ----------------------------------------------------------------------------
char const*
get_env( std::string const& name )
{
  return std::getenv( name.c_str() );
}

} // namespace vital

} // namespace kwiver
