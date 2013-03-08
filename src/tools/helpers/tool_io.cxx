/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "tool_io.h"

#include <boost/filesystem/fstream.hpp>

#include <iostream>

namespace
{

static vistk::path_t const iostream_path = vistk::path_t("-");

}

static void std_stream_dtor(void* ptr);

istream_t
open_istream(vistk::path_t const& path)
{
  istream_t istr;

  if (path == iostream_path)
  {
    istr.reset(&std::cin, &std_stream_dtor);
  }
  else
  {
    istr.reset(new boost::filesystem::ifstream(path));

    if (!istr->good())
    {
      std::string const str = path.string<std::string>();
      std::string const reason = "Unable to open input file: " + str;

      throw std::runtime_error(reason);
    }
  }

  return istr;
}

ostream_t
open_ostream(vistk::path_t const& path)
{
  ostream_t ostr;

  if (path == iostream_path)
  {
    ostr.reset(&std::cout, &std_stream_dtor);
  }
  else
  {
    ostr.reset(new boost::filesystem::ofstream(path));

    if (!ostr->good())
    {
      std::string const str = path.string<std::string>();
      std::string const reason = "Unable to open input file: " + str;

      throw std::runtime_error(reason);
    }
  }

  return ostr;
}

void
std_stream_dtor(void* /*ptr*/)
{
  // We don't want to delete std::cin or std::cout.
}
