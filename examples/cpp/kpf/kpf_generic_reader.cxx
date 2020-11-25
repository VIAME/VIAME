// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 *
 * Generic KPF parser
 *
 */

#include <arrows/kpf/yaml/kpf_reader.h>
#include <arrows/kpf/yaml/kpf_yaml_parser.h>
#include <arrows/kpf/yaml/kpf_yaml_writer.h>

#include <iostream>
#include <fstream>

using std::cerr;
using std::cout;
using std::ifstream;

namespace KPF=kwiver::vital::kpf;

int main( int argc, char *argv[] )
{
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " kpf-yaml-file\n";
    return EXIT_FAILURE;
  }

  ifstream is( argv[1] );
  if ( ! is )
  {
    cerr << "Couldn't open '" << argv[1] << "'\n";
    return EXIT_FAILURE;
  }

  KPF::kpf_yaml_parser_t parser( is );
  KPF::kpf_reader_t reader ( parser );

  size_t line_c = 0;
  while (reader.next())
  {
    ++line_c;
    for (const auto& p: reader.get_meta_packets() )
    {
      cout << "line " << line_c << ": meta '" << p << "'\n";
    }
    for (const auto& p: reader.get_packet_buffer() )
    {
      cout << "line " << line_c << ": " << p.second << "\n";
    }
    reader.flush();
  }
}
