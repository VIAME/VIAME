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
 * \brief Implementation of detected object set csv outputuuid factory
 */

#include "uuid_factory_core.h"

#include <uuid/uuid.h>


namespace kwiver {
namespace arrows {
namespace core {

// ------------------------------------------------------------------
class uuid_factory_core::priv
{
public:



};



// ==================================================================
uuid_factory_core::
uuid_factory_core()
  : d( new uuid_factory_core::priv() )
{

}

uuid_factory_core::
~uuid_factory_core()
{ }


// ------------------------------------------------------------------
void
uuid_factory_core::
set_configuration(vital::config_block_sptr config)
{
}


// ------------------------------------------------------------------
bool
uuid_factory_core::
check_configuration(vital::config_block_sptr config) const
{
  return true;
}


// ------------------------------------------------------------------
  kwiver::vital::uid
uuid_factory_core::
create_uuid()
{
  // This may need work to be more system independent.
  uuid_t new_uuid;
  uuid_generate( new_uuid );
  const char* cc = (const char *)&new_uuid[0];

  return kwiver::vital::uid( cc, sizeof( new_uuid ));
}


} } } // end namespace
