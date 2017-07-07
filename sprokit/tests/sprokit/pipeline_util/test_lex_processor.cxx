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

#include <test_common.h>

#include <sprokit/pipeline_util/lex_processor.h>
#include <vital/vital_types.h>

#include <iostream>
#include <sstream>

#define TEST_ARGS (kwiver::vital::path_t const& pipe_file)

DECLARE_TEST_MAP();

static std::string const pipe_ext = ".pipe";

int
main( int argc, char* argv[] )
{
  CHECK_ARGS( 2 );

  testname_t const testname = argv[1];
  kwiver::vital::path_t const pipe_dir = argv[2];

  kwiver::vital::path_t const pipe_file = pipe_dir + "/" + testname + pipe_ext;

  RUN_TEST( testname, pipe_file );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( lex_pipeline )
{
  struct ground_truth_t
  {
    int type;
    std::string value;
  };

  ground_truth_t truth[] =
  {
    { sprokit::TK_PROCESS,       "process"        },
    { sprokit::TK_IDENTIFIER,    "src1"           },
    { sprokit::TK_DOUBLE_COLON,  "::"             },
    { sprokit::TK_IDENTIFIER,    "numbers"        },
    { sprokit::TK_COLON,         ":"              },
    { sprokit::TK_IDENTIFIER,    "foo"            },
    { sprokit::TK_IDENTIFIER,    "value"          },
    { sprokit::TK_COLON,         ":"              },
    { sprokit::TK_IDENTIFIER,    "foo2"           },
    { ':',                       ":"              },
    { sprokit::TK_IDENTIFIER,    "more"           },
    { sprokit::TK_IDENTIFIER,    "value"          },
    { '.',                       "."              },
    { sprokit::TK_IDENTIFIER,    "a"              },
    { sprokit::TK_COLON,         ":"              },
    { sprokit::TK_IDENTIFIER,    "foo2"           },
    { ':',                       ":"              },
    { sprokit::TK_IDENTIFIER,    "more2"          },
    { sprokit::TK_IDENTIFIER,    "value"          },
    { ':',                       ":"              },
    { sprokit::TK_IDENTIFIER,    "a"              },
    { sprokit::TK_RELATIVE_PATH, "relativepath"   },
    { sprokit::TK_IDENTIFIER,    "foo"            },
    { '=',                       "="              },
    { sprokit::TK_IDENTIFIER,    "bar"            },
    { sprokit::TK_PROCESS,       "process"        },
    { sprokit::TK_IDENTIFIER,    "src2"           },
    { sprokit::TK_DOUBLE_COLON,  "::"             },
    { sprokit::TK_IDENTIFIER,    "numbers"        },
    { sprokit::TK_PROCESS,       "process"        },
    { sprokit::TK_IDENTIFIER,    "end"            },
    { sprokit::TK_DOUBLE_COLON,  "::"             },
    { sprokit::TK_IDENTIFIER,    "multiplication" },
    { sprokit::TK_IDENTIFIER,    "key"            },
    { '=',                       "="              },
    { sprokit::TK_IDENTIFIER,    "value"          },
    { sprokit::TK_IDENTIFIER,    "ket1"           },
    { '=',                       "="              },
    { sprokit::TK_IDENTIFIER,    "value1"         },
    { sprokit::TK_PROCESS,       "process"        },
    { sprokit::TK_IDENTIFIER,    "sink"           },
    { sprokit::TK_DOUBLE_COLON,  "::"             },
    { sprokit::TK_IDENTIFIER,    "sink"           },
    { sprokit::TK_CONNECT,       "connect"        },
    { sprokit::TK_FROM,          "from"           },
    { sprokit::TK_IDENTIFIER,    "src1"           },
    { '.',                       "."              },
    { sprokit::TK_IDENTIFIER,    "number"         },
    { sprokit::TK_TO,            "to"             },
    { sprokit::TK_IDENTIFIER,    "end"            },
    { '.',                       "."              },
    { sprokit::TK_IDENTIFIER,    "factor1"        },
    { sprokit::TK_EOL,           ""               },
    { sprokit::TK_CONFIG,        "config"         },
    { sprokit::TK_IDENTIFIER,    "myblock"        },
    { sprokit::TK_COLON,         ":"              },
    { sprokit::TK_IDENTIFIER,    "mykey"          },
    { sprokit::TK_IDENTIFIER,    "myvalue"        },
    { sprokit::TK_PROCESS,       "process"        },
    { sprokit::TK_IDENTIFIER,    "src1"           },
    { sprokit::TK_DOUBLE_COLON,  "::"             },
    { sprokit::TK_IDENTIFIER,    "numbers"        },
    { sprokit::TK_COLON,         ":"              },
    { sprokit::TK_IDENTIFIER,    "foo"            },
    { sprokit::TK_IDENTIFIER,    "value"          },
    { sprokit::TK_COLON,         ":"              },
    { sprokit::TK_IDENTIFIER,    "foo2"           },
    { ':',                       ":"              },
    { sprokit::TK_IDENTIFIER,    "more"           },
    { sprokit::TK_IDENTIFIER,    "value"          },
    { '.',                       "."              },
    { sprokit::TK_IDENTIFIER,    "a"              },
    { sprokit::TK_COLON,         ":"              },
    { sprokit::TK_IDENTIFIER,    "foo2"           },
    { ':',                       ":"              },
    { sprokit::TK_IDENTIFIER,    "more2"          },
    { sprokit::TK_IDENTIFIER,    "value"          },
    { ':',                       ":"              },
    { sprokit::TK_IDENTIFIER,    "a"              },
    { sprokit::TK_RELATIVE_PATH, "relativepath"   },
    { sprokit::TK_IDENTIFIER,    "foo"            },
    { '=',                       "="              },
    { sprokit::TK_IDENTIFIER,    "bar"            },
    { sprokit::TK_PROCESS,       "process"        },
    { sprokit::TK_IDENTIFIER,    "src2"           },
    { sprokit::TK_DOUBLE_COLON,  "::"             },
    { sprokit::TK_IDENTIFIER,    "numbers"        },
    { sprokit::TK_PROCESS,       "process"        },
    { sprokit::TK_IDENTIFIER,    "end"            },
    { sprokit::TK_DOUBLE_COLON,  "::"             },
    { sprokit::TK_IDENTIFIER,    "multiplication" },
    { sprokit::TK_IDENTIFIER,    "key"            },
    { '=',                       "="              },
    { sprokit::TK_IDENTIFIER,    "value"          },
    { sprokit::TK_IDENTIFIER,    "ket1"           },
    { '=',                       "="              },
    { sprokit::TK_IDENTIFIER,    "value1"         },
    { sprokit::TK_PROCESS,       "process"        },
    { sprokit::TK_IDENTIFIER,    "sink"           },
    { sprokit::TK_DOUBLE_COLON,  "::"             },
    { sprokit::TK_IDENTIFIER,    "sink"           },
    { sprokit::TK_CONNECT,       "connect"        },
    { sprokit::TK_FROM,          "from"           },
    { sprokit::TK_IDENTIFIER,    "src1"           },
    { '.',                       "."              },
    { sprokit::TK_IDENTIFIER,    "number"         },
    { sprokit::TK_TO,            "to"             },
    { sprokit::TK_IDENTIFIER,    "end"            },
    { '.',                       "."              },
    { sprokit::TK_IDENTIFIER,    "factor1"        },
    { sprokit::TK_EOL,           ""               },
    { sprokit::TK_CONFIG,        "config"         },
    { sprokit::TK_IDENTIFIER,    "myblock"        },
    { sprokit::TK_COLON,         ":"              },
    { sprokit::TK_IDENTIFIER,    "mykey"          },
    { sprokit::TK_IDENTIFIER,    "myvalue"        },
    { sprokit::TK_EOL,           ""               },
    { sprokit::TK_CONFIG,        "config"         },
    { sprokit::TK_IDENTIFIER,    "myblock"        },
    { sprokit::TK_COLON,         ":"              },
    { sprokit::TK_IDENTIFIER,    "mykey"          },
    { sprokit::TK_IDENTIFIER,    "myvalue"        },
    { sprokit::TK_IDENTIFIER,    "include"        },
    { sprokit::TK_CONNECT,       "connect"        },
    { sprokit::TK_FROM,          "from"           },
    { sprokit::TK_IDENTIFIER,    "src2"           },
    { '.',                       "."              },
    { sprokit::TK_IDENTIFIER,    "number"         },
    { sprokit::TK_TO,            "to"             },
    { sprokit::TK_IDENTIFIER,    "end"            },
    { '.',                       "."              },
    { sprokit::TK_IDENTIFIER,    "factor2"        },
    { sprokit::TK_CONNECT,       "connect"        },
    { sprokit::TK_FROM,          "from"           },
    { sprokit::TK_IDENTIFIER,    "end"            },
    { '.',                       "."              },
    { sprokit::TK_IDENTIFIER,    "product"        },
    { sprokit::TK_TO,            "to"             },
    { sprokit::TK_IDENTIFIER,    "sink"           },
    { '.',                       "."              },
    { sprokit::TK_IDENTIFIER,    "sink"           },
    { sprokit::TK_EOF,           "E-O-F"          }
  };

  sprokit::lex_processor lex;

  lex.open_file( pipe_file );

  int idx = 0;
  const int limit( sizeof (truth) / sizeof (ground_truth_t) );
  sprokit::token_sptr t;
  do
  {
    t = lex.get_token();
    // std::cout << *t << std::endl;

    if ( idx >= limit )
    {
      TEST_ERROR( "found more tokens than expected" );
      break;
    }

    {
      std::stringstream label;
      label << "token " << idx << " type";
      TEST_EQUAL( label.str(), t->token_value(), truth[idx].type );
    }

    {
      std::stringstream label;
      label << "token " << idx << " value";
      TEST_EQUAL( label.str(), t->text(), truth[idx].value );
    }

    idx++;
  }
  while ( t->token_type() != sprokit::TK_EOF );
}
