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

#include "db_connection.h"


namespace kwiver {
namespace arrows {
namespace database {

class db_connection::priv

{
public:

  /// Constructor
  priv()
    : connect_string_(""),
      is_connected_(false)
  {
  }
  std::string connect_string_;
  bool is_connected_;
  cppdb::session raw_connection_;

};

db_connection::db_connection(std::string conn_str)
  : d_(new priv)
{

  /* kwiver connection string is of the following format
     host=db_host;user=db_user;password=db_pass;dbname=db_name;port=db_port
   */
  d_->connect_string_ += ( "postgresql:" );
#ifdef MODULE_PATH
  d_->connect_string_ += "@modules_path="  MODULE_PATH;
#endif
  d_->connect_string_ += ( ";@blob=bytea");
  d_->connect_string_ += ";" + conn_str;

    /*
      connect_string_ += ( ";host=" + db_host);
      connect_string_ += ( ";user=" + db_user);
      connect_string_ += ( ";password=" + db_pass);
      connect_string_ += ( ";dbname=" + db_name);
      connect_string_ += ( ";port=" + db_port);
    */
}

db_connection::~db_connection()
{

}

bool db_connection::connect()
{
  d_->raw_connection_.open( d_->connect_string_ );
  d_->is_connected_ = true;
  return d_->is_connected_;
}

void db_connection::close_connection()
{
  d_->raw_connection_.close();
  d_->is_connected_ = false;
}

bool db_connection::is_connected()
{
  return d_->is_connected_;
}


} } } // end namespace
