/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief test the kwiver_database
 */

#include <test_common.h>
#include <string>
#include <iostream>
//#include <arrows/database/connection/db_connection.h>
#include <arrows/database/descriptor_db.h>
#include <vital/types/descriptor.h>


#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(descriptor_database)
{

  std::string db_host = "ceres";
  std::string db_user = "perseas";
  std::string db_pass = "";
  std::string db_name = "perseas";
  std::string db_port = "12350";


  std::string conn_str = "host=" + db_host + ";user=" + db_user + ";password=" + db_pass;
  conn_str += ";dbname=" + db_name + ";port=" + db_port;

  kwiver::arrows::database::descriptor_db desc_db(conn_str);
  std::size_t data_size = 5;

  double data[5];
  for (int i = 0; i < data_size; ++i)
  {
    double f = (double)rand() / RAND_MAX;
    data[i] =  0 + f * (10 - 0);
  }

  // = {5.323234, 0.2341234, 1.345345, 0.347345, 0.4575673};
  kwiver::vital::descriptor_sptr desc =
    std::make_shared<kwiver::vital::descriptor_dynamic<double> >(5, data);

  desc_db.add_descriptor(desc);
  desc_db.get_descriptor();

  /*
  kwiver::arrows::database::db_connection db_conn(conn_str);
  bool is_connected = db_conn.is_connected();
  std::cerr << "is_connected: " << is_connected << std::endl;

  TEST_EQUAL( "is db connected", is_connected, false);

  TEST_EQUAL( "empty config", db_conn.connect(), true );
  TEST_EQUAL( "empty config", db_conn.is_connected(), true);

  db_conn.close_connection();
  TEST_EQUAL( "empty config", db_conn.is_connected(), false);
  */
}
