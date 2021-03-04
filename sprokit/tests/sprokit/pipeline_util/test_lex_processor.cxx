// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
