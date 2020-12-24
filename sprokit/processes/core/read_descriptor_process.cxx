// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Accept vector of doubles from descriptor.
 */

#include "read_descriptor_process.h"

#include <vital/vital_types.h>

#include <kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver
{

// should be promoted to project level include
create_port_trait( d_vector, double_vector, "Vector of doubles from descriptor" );

// config items
//          None for now

//----------------------------------------------------------------
// Private implementation class
class read_descriptor_process::priv
{
public:
  priv();
  ~priv();

  // empty for now
};

// ===============================================================================

read_descriptor_process
::read_descriptor_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new read_descriptor_process::priv )
{
  make_ports();
  make_config();
}

read_descriptor_process
::~read_descriptor_process()
{
}

// -------------------------------------------------------------------------------
void
read_descriptor_process
::_configure()
{
}

// -------------------------------------------------------------------------------
void
read_descriptor_process
::_step()
{
  kwiver::vital::double_vector_sptr vect = grab_from_port_using_trait( d_vector );

  {
    scoped_step_instrumentation();

    std::cout << "Vector size: " << vect->size() << " -- " << std::endl;

    for (int i = 0; i < 50; i++)
    {
      std::cout << " " << vect->at(i);
    }
    std::cout << std::endl;
  }
}

// -------------------------------------------------------------------------------
void
read_descriptor_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( d_vector, required );
}

// -------------------------------------------------------------------------------
void
read_descriptor_process
::make_config()
{
}

// ===============================================================================
read_descriptor_process::priv
::priv()
{
}

read_descriptor_process::priv
::~priv()
{
}

} // end namespace
