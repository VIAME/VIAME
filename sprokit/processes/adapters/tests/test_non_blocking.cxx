// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_common.h>

#include <vital/config/config_block.h>

#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_factory.h>
#include <sprokit/processes/adapters/output_adapter.h>
#include <sprokit/processes/adapters/output_adapter_process.h>
#include <sprokit/processes/adapters/embedded_pipeline.h>
#include <sprokit/pipeline_util/literal_pipeline.h>

#include <sstream>
#include <thread>

class test_non_blocking
  : public sprokit::process
{
public:
  test_non_blocking(kwiver::vital::config_block_sptr const& config)
    : sprokit::process( config )
  {
    port_flags_t optional;
    port_flags_t required;
    required.insert(flag_required);

    // make "input" port
    this->declare_input_port(
      "input",
      "integer",
      required,
      port_description_t("Where the numbers will be available."));

    // make "output" port
    this->declare_output_port(
      "output",
      "integer",
      required,
      port_description_t("Where the numbers will be available."));

  }

  ~test_non_blocking()
  { }

protected:
  void _configure()
  {

  }

  void _step()
  {
    sprokit::datum_t dat;

    dat = grab_datum_from_port( "input" );
    std::cout << "----- received data value: " << dat->get_datum<int>() << std::endl;
    // simulate a small amount of processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    push_datum_to_port( "output", dat );
  }

};

// ==================================================================
#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  //
  // Register local process(es)
  //

  auto& vpm = kwiver::vital::plugin_manager::instance();
  vpm.load_all_plugins();

  // Register local process
  auto fact = vpm.ADD_PROCESS( test_non_blocking );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "test_non_blocking" );

  RUN_TEST(testname);
}

// ==================================================================
IMPLEMENT_TEST(nonblocking)
{
  std::stringstream pipeline_desc;
  pipeline_desc
    << SPROKIT_PROCESS( "input_adapter",  "ia" )
    << SPROKIT_PROCESS( "output_adapter", "oa" )

    << SPROKIT_PROCESS( "test_non_blocking", "non_b" )
    << SPROKIT_CONFIG( "_non_blocking", "1" )

    << SPROKIT_CONNECT( "ia",       "number",        "non_b", "input" )
    << SPROKIT_CONNECT( "non_b",    "output",        "oa",    "number" )

    << SPROKIT_CONFIG_BLOCK( "_pipeline:_edge")
    << SPROKIT_CONFIG( "capacity",  "300" )
    ;

  // create embedded pipeline
  kwiver::embedded_pipeline ep;
  ep.build_pipeline( pipeline_desc );

  // Start pipeline
  ep.start();
  constexpr int limit(50);

  for ( int i = 0; i < limit; ++i )
  {
    auto ds = kwiver::adapter::adapter_data_set::create();
    ds->add_value( "number", i );

    // inject the data in bursts of 5 frames with a short delay between
    if ( (i % 5) == 0 )
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "sending set: " << i << "\n";
    ep.send( ds );
  }

  //+ std::cout << "Sending end of input element\n";
  ep.send_end_of_input();

  std::set<int> received;

  while (true)
  {
    auto rds = ep.receive(); // Retrieve end of input data item.
    if ( ep.at_end() )
    {
      break;
    }

    // Get value from output adapter
    int num = rds->get_port_data<int>("number");
    received.insert( num );
    //+ std::cout << "received: " << num << std::endl;
  }

  ep.wait();

  int dcount(0);

  for ( int i = 0; i < limit; ++i )
  {
    if ( 0 == received.count( i ) )
    {
      std::cout << "Data value " << i << " dropped in the pipeline" << std::endl;
      ++dcount;
    }
  } // end for

  if ( 0 == dcount )
  {
    TEST_ERROR( "Expected some elements to be dropped but found none." );
  }
}
