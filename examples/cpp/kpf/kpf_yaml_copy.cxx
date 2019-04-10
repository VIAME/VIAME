/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
