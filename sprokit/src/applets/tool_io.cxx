// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "tool_io.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

namespace sprokit {

namespace {

static kwiver::vital::path_t const iostream_path = kwiver::vital::path_t("-");

}

static void std_stream_dtor(void* ptr);

// ------------------------------------------------------------------
istream_t
open_istream(kwiver::vital::path_t const& path)
{
  istream_t istr;

  if (path == iostream_path)
  {
    istr.reset(&std::cin, &std_stream_dtor);
  }
  else
  {
    istr.reset(new std::ifstream(path));

    if (!istr->good())
    {
      std::string const reason = "Unable to open input file: " + path;

      throw std::runtime_error(reason);
    }
  }

  return istr;
}

// ------------------------------------------------------------------
ostream_t
open_ostream(kwiver::vital::path_t const& path)
{
  ostream_t ostr;

  if (path == iostream_path)
  {
    ostr.reset(&std::cout, &std_stream_dtor);
  }
  else
  {
    ostr.reset(new std::ofstream(path));

    if (!ostr->good())
    {
      std::string const reason = "Unable to open output file: " + path;

      throw std::runtime_error(reason);
    }
  }

  return ostr;
}

// ------------------------------------------------------------------
void
std_stream_dtor(void* /*ptr*/)
{
  // We don't want to delete std::cin or std::cout.
}

} // end namespace
