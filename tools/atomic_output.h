// This file is part of VIAME, distributed under the BSD 3-Clause License.
#ifndef VIAME_TOOLS_ATOMIC_OUTPUT_H
#define VIAME_TOOLS_ATOMIC_OUTPUT_H

#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

namespace viame { namespace tools {

// Stage beside the destination so rename stays on one filesystem. An
// exclusively created directory isolates simultaneous writers and symlinks.
template <typename Write>
void atomic_output( const std::string& filename, Write write )
{
  namespace fs = std::filesystem;
  const fs::path destination = fs::absolute( filename );
  fs::path staging;
  std::random_device random;
  for( unsigned attempt = 0; attempt < 100; ++attempt )
  {
    auto candidate = destination.parent_path() /
      (".viame-write-" + std::to_string( random() ) + "-" + std::to_string( random() ));
    if( fs::create_directory( candidate ) )
    {
      staging = candidate;
      break;
    }
  }
  if( staging.empty() )
  {
    throw std::runtime_error( filename + ": could not create staging directory" );
  }
  try
  {
    const fs::path temporary = staging / "output";
    std::ofstream out;
    out.exceptions( std::ios::failbit | std::ios::badbit );
    out.open( temporary, std::ios::binary );
    write( out );
    out.flush();
    out.close();
    if( fs::exists( destination ) )
    {
      fs::permissions( temporary, fs::status( destination ).permissions() );
    }
    fs::rename( temporary, destination );
  }
  catch( ... )
  {
    std::error_code ignored;
    fs::remove_all( staging, ignored );
    throw;
  }
  std::error_code ignored;
  fs::remove_all( staging, ignored );
}

} }
#endif
