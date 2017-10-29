//
// A simple example that reads KPFs.
//

#include <vital/kpf/kpf_parse.h>
#include <vital/kpf/kpf_text_parser.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace KPF=kwiver::vital::kpf;


int main( int argc, char *argv[] )
{
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " somefile.kpf\n"
              << "Reads and reports on a KPF file.\n";
    return EXIT_SUCCESS;
  }

  std::ifstream is( argv[1] );
  if ( ! is )
  {
    std::cerr << "Couldn't open '" << argv[1] << "' for reading\n";
    return EXIT_FAILURE;
  }


  KPF::kpf_text_parser_t parser( is );
  KPF::kpf_reader_t reader( parser );
  while (reader.next())
  {
    const KPF::packet_buffer_t& packets = reader.get_packet_buffer();

    std::vector< std::string > meta = reader.get_meta_packets();
    std::cout << "Parsed " << meta.size() << " metadata packets:\n";
    for (auto m: meta)
    {
      std::cout << "== " << m << "\n";
    }
    std::cout << "Parsed " << packets.size() << " packets:\n";
    for (auto p: packets )
    {
      std::cout << "-- " << p.second << "\n";
    }
    reader.flush();
  }
}
