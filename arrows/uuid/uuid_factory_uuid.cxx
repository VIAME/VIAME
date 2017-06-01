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
 * \brief Implementation of uuid factory
 */

#include "uuid_factory_uuid.h"

#if defined WIN32
#include <windows.h>
#else
#include <uuid/uuid.h>
#endif

namespace kwiver {
namespace arrows {
namespace uuid {

// ------------------------------------------------------------------
class uuid_factory_uuid::priv
{
public:

};


// ==================================================================
uuid_factory_uuid::
uuid_factory_uuid()
  : d( new uuid_factory_uuid::priv() )
{

}


uuid_factory_uuid::
~uuid_factory_uuid()
{ }


// ------------------------------------------------------------------
void
uuid_factory_uuid::
set_configuration(vital::config_block_sptr config)
{
}


// ------------------------------------------------------------------
bool
uuid_factory_uuid::
check_configuration(vital::config_block_sptr config) const
{
  return true;
}


// ------------------------------------------------------------------
kwiver::vital::uid
uuid_factory_uuid::
create_uuid()
{
#if defined WIN32
  UUID new_uuid;

  UuidCreate( &new_uuid );

  const char* cc = (const char*) &new_uuid.Data4[0];
#else
  // This may need work to be more system independent.
  uuid_t new_uuid;

  // from libuuid
  uuid_generate( new_uuid );
  const char* cc = (const char *) &new_uuid[0];
#endif

  return kwiver::vital::uid( cc, sizeof( new_uuid ));
}


} } } // end namespace
