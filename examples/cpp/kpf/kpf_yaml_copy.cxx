// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief A simple demonstration that reads a yaml KPF file and writes it back out.
 *
 */

#include <arrows/kpf/yaml/kpf_reader.h>
#include <arrows/kpf/yaml/kpf_yaml_parser.h>
#include <arrows/kpf/yaml/kpf_yaml_writer.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace KPF=kwiver::vital::kpf;

int main( int argc, char* argv[] )
{
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " file.kpf\n";
    return EXIT_FAILURE;
  }

  std::ifstream is( argv[1] );
  if (!is)
  {
    std::cerr << "Couldn't open '" << argv[1] << "' for reading\n";
    return EXIT_FAILURE;
  }

  KPF::kpf_yaml_parser_t parser( is );
  KPF::kpf_reader_t reader( parser );
  KPF::record_yaml_writer writer( std::cout );
  while (reader.next())
  {
    const KPF::packet_buffer_t& packets = reader.get_packet_buffer();

    std::vector< std::string > meta = reader.get_meta_packets();
    writer.set_schema( KPF::schema_style::META );
    for (auto m: meta)
    {
      writer << m << KPF::record_yaml_writer::endl;
    }
    writer.set_schema( parser.get_current_record_schema() );
    for (auto p: packets )
    {
      writer << p.second;
    }
    writer << KPF::record_yaml_writer::endl;
    reader.flush();
  }
}
