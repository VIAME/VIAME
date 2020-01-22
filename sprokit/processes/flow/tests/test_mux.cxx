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

#include <sprokit/processes/adapters/embedded_pipeline.h>
#include <sprokit/pipeline_util/literal_pipeline.h>

#include <sstream>


#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


// ==================================================================
// This class removes the requirement for a input adapter in the
// pipeline.
class src_ep
  : public kwiver::embedded_pipeline
{
public:
  src_ep() = default;
  virtual ~src_ep() = default;

protected:
  virtual bool connect_input_adapter() { return true; }
};


// ------------------------------------------------------------------
IMPLEMENT_TEST( test_mux )
{
  // Use SPROKIT macros to create pipeline description
  std::stringstream pipeline_desc;
  pipeline_desc
    << SPROKIT_PROCESS( "numbers", "num_1" )
    << SPROKIT_CONFIG( "start", "0" )
    << SPROKIT_CONFIG( "end",   "20" )

    << SPROKIT_PROCESS( "numbers", "num_2" )
    << SPROKIT_CONFIG( "start", "100" )
    << SPROKIT_CONFIG( "end",   "120" )

    << SPROKIT_PROCESS( "numbers", "num_3" )
    << SPROKIT_CONFIG( "start", "200" )
    << SPROKIT_CONFIG( "end",   "220" )

    << SPROKIT_PROCESS( "multiplexer",  "mux" )
    << SPROKIT_CONFIG( "termination_policy",   "all" )

    << SPROKIT_CONNECT( "num_1", "number",     "mux", "num/1" )
    << SPROKIT_CONNECT( "num_2", "number",     "mux", "num/2" )
    << SPROKIT_CONNECT( "num_3", "number",     "mux", "num/3" )
    << SPROKIT_CONNECT( "num_2", "number",     "mux", "num2/2" )
    << SPROKIT_CONNECT( "num_1", "number",     "mux", "num2/3" )
    << SPROKIT_CONNECT( "num_3", "number",     "mux", "num2/1" )


    << SPROKIT_PROCESS( "output_adapter", "oa" )

    << SPROKIT_CONNECT( "mux", "num",  "oa", "num" )
    << SPROKIT_CONNECT( "mux", "num2", "oa", "num2" )
    ;

  // expected output values
  int num_expected[60] = {
    0, 100, 200, 1, 101, 201, 2, 102, 202, 3, 103, 203, 4, 104, 204, 5, 105, 205,
    6, 106, 206, 7, 107, 207, 8, 108, 208, 9, 109, 209, 10, 110, 210, 11, 111, 211,
    12, 112, 212, 13, 113, 213, 14, 114, 214, 15, 115, 215, 16, 116, 216, 17, 117, 217,
    18, 118, 218, 19, 119, 219
  };

  int num2_expected[60] = {
    200, 100, 0, 201, 101, 1, 202, 102, 2, 203, 103, 3, 204, 104, 4, 205, 105, 5,
    206, 106, 6, 207, 107, 7, 208, 108, 8, 209, 109, 9, 210, 110, 10, 211, 111, 11,
    212, 112, 12, 213, 113, 13, 214, 114, 14, 215, 115, 15, 216, 116, 16, 217, 117,
    17, 218, 118, 18, 219, 119, 19
  };

  // --------------------------
  // create embedded pipeline
  src_ep ep;
  ep.build_pipeline( pipeline_desc );

  // Start pipeline
  ep.start();

  int idx(0);

  while( true )
  {
    // Get output
    auto ods = ep.receive(); // blocks

    // check for end of data marker
    if ( ods->is_end_of_data() )
    {
      TEST_EQUAL( "at_end() set correctly", ep.at_end(), true );
      break;
    }

    int val = ods->get_port_data<int>( "num" );
    int val2 = ods->get_port_data<int>( "num2" );

    if ( val != num_expected[idx] )
    {
      std::stringstream str;
      str << "Unexpected value from port num. Expected " << num_expected[idx]
          << " got " << val;
      TEST_ERROR( str.str() );
    }

    if ( val2 != num2_expected[idx] )
    {
      std::stringstream str;
      str << "Unexpected value from port num2. Expected " << num2_expected[idx]
          << " got " << val2;
      TEST_ERROR( str.str() );
    }

    // increment to next input
    ++idx;
  } // end while

  ep.wait(); // wait for pipeline to terminate
}
