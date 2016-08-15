/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <track_oracle/utils/tokenizers.h>

#include <vital/logger/logger.h>

using std::istringstream;
using std::string;
using std::vector;

static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

int main( int argc, char *argv[] )
{
  if (argc != 3)
  {
    LOG_INFO( main_logger, "Usage: " << argv[0] << " xml-file  n-tokens");
    return EXIT_FAILURE;
  }

  istringstream iss( argv[2] );
  size_t n;
  if ( ! (iss >> n ))
  {
    LOG_ERROR( main_logger, "Couldn't extract n-tokens from '" << argv[2] << "'");
    return EXIT_FAILURE;
  }

  vector< string > tokens = kwiver::track_oracle::xml_tokenizer::first_n_tokens( argv[1], n );
  LOG_INFO( main_logger, "Got " << tokens.size() << " tokens:");
  for (size_t i=0; i<tokens.size(); ++i)
  {
    LOG_INFO( main_logger, i << ":\t '" << tokens[i] << "'");
  }
}
