
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

#ifndef __ARROWS_DATATBASE_DESCRIPTOR_DB_H__
#define __ARROWS_DATATBASE_DESCRIPTOR_DB_H__

#include <arrows/database/kwiver_algo_database_export.h>
#include <arrows/database/connection/db_connection.h>

#include <vital/types/descriptor.h>
#include <vital/types/descriptor_set.h>

#include <cppdb/frontend.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace database {

class KWIVER_ALGO_DATABASE_EXPORT descriptor_db
{

public:

  descriptor_db( std::string conn_str );
  virtual ~descriptor_db();

  bool add_descriptor( kwiver::vital::descriptor_sptr const desc );
  bool add_descriptor_set( kwiver::vital::descriptor_set_sptr const desc_set );
  kwiver::vital::descriptor_sptr get_descriptor( );

private:

  cppdb::session db_conn_;
  std::string connect_string_;

  //db_connection db_conn_;

}; // class descriptor_db

} } } // end namespace

#endif // __ARROWS_DATATBASE_DESCRIPTOR_DB_H__
