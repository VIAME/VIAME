/*ckwg +29
 * Copyright 2013-2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
