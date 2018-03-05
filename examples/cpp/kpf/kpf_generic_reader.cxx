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
