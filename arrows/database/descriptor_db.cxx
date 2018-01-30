
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

#include "descriptor_db.h"
#include <arrows/database/descriptor_db_defs.h>

#include <iomanip>

namespace kwiver {
namespace arrows {
namespace database {

descriptor_db
::descriptor_db( std::string conn_str )

{
    /* kwiver connection string is of the following format
     host=db_host;user=db_user;password=db_pass;dbname=db_name;port=db_port
   */
  connect_string_ += ( "postgresql:" );
#ifdef MODULE_PATH
  connect_string_ += "@modules_path="  MODULE_PATH;
#endif
  connect_string_ += ( ";@blob=bytea");
  connect_string_ += ";" + conn_str;

}

descriptor_db
::~descriptor_db()
{

}

bool
descriptor_db::add_descriptor( kwiver::vital::descriptor_sptr const desc )
{
  db_conn_.open( connect_string_ );

  std::size_t desc_size = desc->size();
  //std::vector< kwiver::vital::byte > bytes = desc->as_bytes();
  std::vector<double> data = desc->as_double();
  //std::size_t num_bytes = desc->num_bytes();


  std::stringstream data_str;
  data_str << "'{";

  std::vector<double>::const_iterator iter = data.begin();
  bool first = true;
  for (; iter != data.end(); ++iter)
  {
    if  (!first)
    {
      data_str << ",";
    }
    first = false;
    double d = *iter;
    data_str << std::fixed << std::setprecision( 20 ) << d;
  }

  data_str << "}'";

  std::string query = "insert into descriptor (data_size, data) values (5," + data_str.str() + ")";
  cppdb::transaction guard(db_conn_);
  try
  {
    cppdb::statement stmt = db_conn_.create_statement( query );
    stmt.exec();
    guard.commit();
  }
  catch(cppdb::cppdb_error const &e)
  {
    guard.rollback();
    CLOSE_DB_CONN;
    std::cerr << e.what() << std::endl;
    return false;
  }

  CLOSE_DB_CONN;
  return true;
}


bool
descriptor_db::add_descriptor_set( kwiver::vital::descriptor_set_sptr const desc_set )
{

  return true;
}



kwiver::vital::descriptor_sptr
descriptor_db::get_descriptor( )
{
  OPEN_DB_CONN;
  std::string query = "select data_size, data from descriptor";
  try
  {
    cppdb::statement stmt = db_conn_.create_statement( query );
    cppdb::result rs = stmt.query();
    while (rs.next ())
    {

      int size = rs.get<int>( "data_size");
      std::string data = rs.get<std::string>( "data");
      std::cerr << size << std::endl;
      std::cerr << data << std::endl;
    }
  }
  catch(cppdb::cppdb_error const &e)
  {
    CLOSE_DB_CONN;
    std::cerr << e.what() << std::endl;
    return NULL;
  }

  CLOSE_DB_CONN;

  kwiver::vital::descriptor_sptr desc = std::make_shared<kwiver::vital::descriptor_dynamic<double> >(5);
  return desc;
}

} } } // end namespace
