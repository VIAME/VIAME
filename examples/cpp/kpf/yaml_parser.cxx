/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

/**
 * \file
 * \brief A very simple YAML reader (no KPF).
 *
 */

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <string>

using std::cout;
using std::cerr;
using std::ostream;
using std::ifstream;
using std::string;

string
indent( size_t depth )
{
  string ret;
  for (size_t i=0; i<depth; ++i)
  {
    ret += "  ";
  }
  return ret;
}

void visit( ostream& os, size_t depth, const YAML::Node& n )
{
  //  os << indent( depth );
  switch (n.Type())
  {
  case YAML::NodeType::Null:
    os << "(null)\n";
    break;
  case YAML::NodeType::Scalar:
    os << "(scalar) '" << n.as<string>() << "'\n";
    break;
  case YAML::NodeType::Sequence:
    os << "(sequence)\n";
    for (auto it=n.begin(); it != n.end(); ++it)
    {
      os << indent( depth );
      visit( os, depth+1, *it);
    }
    break;
  case YAML::NodeType::Map:
    os << "(map)\n";
    for (auto it=n.begin(); it != n.end(); ++it)
    {
      os << indent(depth) << it->first.as<string>() << " => ";
      visit( os, depth+1, it->second);
    }
    break;
  case YAML::NodeType::Undefined:
    os << "(undef)\n";
    break;
  default:
    os << "(unhandled " << n.Type() << ")\n";
    break;
  }
}

int main( int argc, char* argv[] )
{
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " file.yml\n";
    return EXIT_FAILURE;
  }

  try
  {
    ifstream is( argv[1] );
    if (!is)
    {
      cerr << "Couldn't open '" << argv[1] << "' for reading\n";
      return EXIT_FAILURE;
    }

    YAML::Node doc = YAML::Load( is );

    for (auto it = doc.begin(); it != doc.end(); ++it)
    {
      visit( cout, 1, *it);
    }
  }
  catch (const YAML::Exception& e )
  {
    cerr << "YAML exception: " << e.what() << "\n";
  }
}
