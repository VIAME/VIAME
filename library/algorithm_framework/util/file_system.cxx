// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief File system operations, over kwiversys for now.
///
/// This is the first half of P8-T05. The call sites move to these names
/// against an implementation that still calls `kwiversys::SystemTools`, so
/// that the recording in
/// `tests/library/algorithm_framework/test_file_system.cxx` is taken against
/// the behaviour the tree has always had. The second half replaces what is
/// below with `std::filesystem` and holds it to that recording.

#include <viame/algorithm_framework/util/file_system.h>

#include <kwiversys/Directory.hxx>
#include <kwiversys/Status.hxx>
#include <kwiversys/SystemTools.hxx>

namespace kwiver {

namespace vital {

namespace {

using ST = kwiversys::SystemTools;

} // namespace

// ----------------------------------------------------------------------------
bool
file_exists( std::string const& path )
{
  return ST::FileExists( path );
}

// ----------------------------------------------------------------------------
bool
file_is_directory( std::string const& path )
{
  return ST::FileIsDirectory( path );
}

// ----------------------------------------------------------------------------
bool
file_is_full_path( std::string const& path )
{
  return ST::FileIsFullPath( path );
}

// ----------------------------------------------------------------------------
std::string
filename_path( std::string const& path )
{
  return ST::GetFilenamePath( path );
}

// ----------------------------------------------------------------------------
std::string
filename_name( std::string const& path )
{
  return ST::GetFilenameName( path );
}

// ----------------------------------------------------------------------------
std::string
filename_last_extension( std::string const& path )
{
  return ST::GetFilenameLastExtension( path );
}

// ----------------------------------------------------------------------------
std::string
filename_without_last_extension( std::string const& path )
{
  return ST::GetFilenameWithoutLastExtension( path );
}

// ----------------------------------------------------------------------------
std::string
parent_directory( std::string const& path )
{
  return ST::GetParentDirectory( path );
}

// ----------------------------------------------------------------------------
void
split_path( std::string const& path, std::vector< std::string >& components )
{
  ST::SplitPath( path, components );
}

// ----------------------------------------------------------------------------
std::string
join_path( std::vector< std::string > const& components )
{
  return ST::JoinPath( components );
}

// ----------------------------------------------------------------------------
std::string
collapse_full_path( std::string const& path )
{
  return ST::CollapseFullPath( path );
}

// ----------------------------------------------------------------------------
std::string
collapse_full_path( std::string const& path, std::string const& base )
{
  return ST::CollapseFullPath( path, base );
}

// ----------------------------------------------------------------------------
std::string
real_path( std::string const& path )
{
  return ST::GetRealPath( path );
}

// ----------------------------------------------------------------------------
std::string
current_working_directory()
{
  return ST::GetCurrentWorkingDirectory();
}

// ----------------------------------------------------------------------------
void
convert_to_unix_slashes( std::string& path )
{
  ST::ConvertToUnixSlashes( path );
}

// ----------------------------------------------------------------------------
bool
make_directory( std::string const& path )
{
  return ST::MakeDirectory( path ).IsSuccess();
}

// ----------------------------------------------------------------------------
bool
remove_directory( std::string const& path )
{
  return ST::RemoveADirectory( path ).IsSuccess();
}

// ----------------------------------------------------------------------------
bool
remove_file( std::string const& path )
{
  return ST::RemoveFile( path ).IsSuccess();
}

// ----------------------------------------------------------------------------
std::string
find_file(
  std::string const& name, std::vector< std::string > const& directories )
{
  return ST::FindFile( name, directories );
}

// ----------------------------------------------------------------------------
std::string
find_program(
  std::string const& name, std::vector< std::string > const& directories )
{
  return ST::FindProgram( name, directories );
}

// ----------------------------------------------------------------------------
std::vector< std::string >
directory_entries( std::string const& path )
{
  std::vector< std::string > names;

  kwiversys::Directory dir;

  if( !dir.Load( path ) )
  {
    return names;
  }

  auto const count = dir.GetNumberOfFiles();

  for( unsigned long i = 0; i < count; ++i )
  {
    names.emplace_back( dir.GetFile( i ) );
  }

  return names;
}

// ----------------------------------------------------------------------------
bool
get_env( std::string const& name, std::string& value )
{
  return ST::GetEnv( name, value );
}

// ----------------------------------------------------------------------------
char const*
get_env( std::string const& name )
{
  return ST::GetEnv( name );
}

} // namespace vital

} // namespace kwiver
