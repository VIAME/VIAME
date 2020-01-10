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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \file mux_process.cxx
 *
 * \brief Implementation of the mux process.
 */

#include "mux_process.h"

#include <vital/util/tokenize.h>
#include <vital/logger/logger.h>

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/edge.h>
#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/stamp.h>

#include <map>
#include <string>
#include <algorithm>

namespace sprokit {

/**
 * \class mux_process
 *
 * \brief A process for collating input data from multiple input edges.
 *
 * \process Multiplex incoming data into a single stream.  A mux
 * operation reads input from a group of input ports and serializes
 * that data to a single output port. The ports in a group are read in
 * ASCII-betical order over the third port name component (item).
 * This mux process can handle multiple collation operations. Each set
 * of collation ports is identified by a unique \b group name.
 *
 * \iports
 *
 * Input ports are dynamically created as needed. Port names have the
 * format \iport{\portvar{group/item}}
 *
 * \oports
 *
 * \oport{\portvar{group}} The multiplexed result \portvar{group}.
 *
 * \reqs
 *
 * \req Each \portvar{group} must have at least two inputs to mux.
 * \req Each output port \port{\portvar{group}} must be connected.
 *
 * This process automatically makes the input and output types for
 * each \b group the same based on the type of the port that is first
 * connected.
 *
 * \code
 process mux :: mux_process

 # -- Connect mux set "group1"
 # All inputs for a group must have the same type
 connect foo_1.out       to  mux.group1/A
 connect foo_2.out       to  mux.group1/B

 # Create another group for the timestamp outputs.
 # For convenience the group name is "timestamp"
 connect foo_1.timestamp to  mux.timestamp/A
 connect foo_2.timestamp to  mux.timestamp/B

 connect mux.group1    to bar.input # connect output
 connect mux.timestamp to bar.timestamp # connect output

 # -- Connect mux set "input2"
 connect foo_1.out       to  mux.input2/A
 connect foo_2.out       to  mux.input2/B
 connect foo_3.out       to  mux.input2/C

 connect mux.input2  to bar.other # connect output

 * \endcode
 *
 * \todo Add configuration to allow forcing a number of inputs for a result.
 * \todo Add configuration to allow same number of sources for all results.
 *
 * \ingroup process_flow
 */

class mux_process::priv
{
public:
  priv();
  ~priv();

  enum term_policy_t { skip, policy_all, policy_any };
  using group_t = port_t;

  // This class stores info for each group.
  class group_info
  {
  public:
    group_info();
    ~group_info();

    ports_t ports; // vector of port names
    ports_t::iterator cur_port;
  };
  using group_data_t = std::map< group_t, group_info >;

  group_data_t group_data; // group table

  static port_t const res_sep;

  term_policy_t m_config_term_policy;
};

process::port_t const mux_process::priv::res_sep = port_t( "/" );

/**
 * \internal
 *
 * Ports on the \ref distribute_process are broken down as follows:
 *
 *   \portvar{type}/\portvar{group}[/\portvar{item}]
 *
 * The port name is broken down as follows:
 *
 * <dl>
 * \term{\portvar{type}}
 *   \termdef{The type of the port. This must be one of \type{res},
 *   or \type{in}.}
 * \term{\portvar{group}}
 *   \termdef{The name of the stream the port is associated with.}
 * \term{\portvar{item}}
 *   \termdef{Only required for \type{in}-type ports. Data from the same
 *   \portvar{group} stream from its \type{res} port is collected in sorted order
 *   over all of the \type{in} ports.}
 * </dl>
 *
 * The available port types are:
 *
 * <dl>
 * \term{\type{res}}
 *   \termdef{This port for the given group is where the data for a stream leaves
 *   the process.
 * \term{\type{in}}
 *   \termdef{These ports for a given \portvar{group} receive data from a set of
 *   sources, likely made by the \ref distribute_process. Data is collected in
 *   sorted ordef of the \type{item} name and sent out the \type{res} port for
 *   the \portvar{group}.}
 * </dl>
 */

mux_process
::mux_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
  d( new priv )
{
  // This process manages its own inputs.
  this->set_data_checking_level( check_none );

  declare_configuration_key( "termination_policy", "any",
                             "Termination policy specifies how a data group is handled when the inputs complete. "
                             "Valid values are \"any\" and \"all\". "
                             "When \"any\" is specified, the output port for the group will complete when any of "
                             "the inputs completes and the remaining active inputs will no longer be polled for data. "
                             "When \"all\" is specified, the output port for the group will complete when all of "
                             "the inputs are complete. "
                           );

}


mux_process
::~mux_process()
{
}


// ----------------------------------------------------------------
void
mux_process
::_configure()
{
  // Examine the configuration
  const std::string value = config_value< std::string > ( "termination_policy" );

  if ( value == "any" )
  {
    d->m_config_term_policy = priv::term_policy_t::policy_any;
  }
  else if ( value == "all" )
  {
    d->m_config_term_policy = priv::term_policy_t::policy_all;
  }
  else
  {
    std::string const reason = "Invalid option specified for termination_policy: " + value;
    VITAL_THROW( invalid_configuration_exception, name(), reason );
  }
}


// ------------------------------------------------------------------
// Post connection processing
void
mux_process
::_init()
{
  for( priv::group_data_t::value_type & group_data : d->group_data )
  {
    priv::group_t const& group = group_data.first;
    priv::group_info& info = group_data.second;
    ports_t const& ports = info.ports; // list of port names

    if ( ports.size() < 2 )
    {
      std::string const reason = "There must be at least two ports to mux "
                                 "to for the \"" + group + "\" result data";

      VITAL_THROW( invalid_configuration_exception, name(), reason );
    }

    // The ports need to be sorted to provide a deterministic order
    // for ports to be processed. Since these are all from the same
    // group, we are really sorting on the third field (item).
    std::sort( info.ports.begin(), info.ports.end() );

    // Now here's some port frequency magic
    frequency_component_t const ratio = ports.size();
    port_frequency_t const freq = port_frequency_t( 1, ratio );

    // Set port frequency for all input ports.
    for( port_t const & port : ports )
    {
      set_input_port_frequency( port, freq );
    }

    // Set iterator to start of list.
    info.cur_port = info.ports.begin();
  }
}


// ------------------------------------------------------------------
void
mux_process
::_reset()
{
  for( const auto & group_data : d->group_data )
  {
    port_t const output_port = group_data.first;

    priv::group_info const& info = group_data.second;
    ports_t const& ports = info.ports;

    for( const auto & port : ports )
    {
      remove_input_port( port );
    }

    remove_output_port( output_port );
  }

  d->group_data.clear();
}


// ------------------------------------------------------------------
void
mux_process
::_step()
{
  ports_t complete_ports;

  // Loop over all groups (input groups) and process the next input
  // port in the group. Even though it may be tempting, only process a
  // single input per step() cycle.
  for ( priv::group_data_t::value_type& group_data : d->group_data )
  {
    priv::group_t const& group = group_data.first;
    port_t const output_port = group;
    priv::group_info& info = group_data.second;

    // There is real data on the input ports. Grab data from the
    // current input port and push to the output.
    edge_datum_t const input_edat = grab_from_port( *info.cur_port );

    LOG_TRACE( logger(), "Fetching from port \"" <<  *info.cur_port << "\"" );

    // check for complete on input port
    datum_t const& input_dat = input_edat.datum;
    datum::type_t const input_type = input_dat->type();

    // Test to see if complete.
    // If the upstream process is done, then mark this group as done.
    if ( input_type == datum::complete )
    {
      LOG_TRACE( logger(),
                 "Data complete on port \"" << *info.cur_port << "\"" );

      // check with termination policy.
      switch ( d->m_config_term_policy )
      {
        case priv::term_policy_t::policy_any:
          // Flush this set of inputs
          for ( port_t const& port : info.ports )
          {
            (void) grab_from_port( port );
          }

          // echo the input control message to the output port
          push_to_port( output_port, input_edat );

          complete_ports.push_back( group );
          break;

        case priv::term_policy_t::policy_all:
        {
          // remove this port only from the "group_data"
          info.cur_port = info.ports.erase( info.cur_port ); // updates iterator

          // need to check for wrapping past end.
          if ( info.cur_port == info.ports.end() )
          {
            info.cur_port = info.ports.begin();
          }

          // If there are no more input ports in this group.
          if ( info.ports.empty() )
          {
            complete_ports.push_back( group );

            // echo the input control message to the output port
            push_to_port( output_port, input_edat );
          }
          break;
        }

        default:
          VITAL_THROW( invalid_configuration_exception, name(),
                       "Invalid option specified for termination_policy." );
      } // end switch

      continue;
    } // end datum::complete

    LOG_TRACE( logger(), "Pushing data to port \"" << output_port << "\"" );

    // Send the input to the output port.
    push_datum_to_port( output_port, input_dat );

    // Advance to next port in the group, and wrap at the end.
    ++info.cur_port;
    if ( info.cur_port == info.ports.end() )
    {
      info.cur_port = info.ports.begin();
    }
  } // end foreach

  // Process all ports/groups that have completed. When a status port
  // reports complete on a group, that group is erased from the local
  // map. When that map is empty, then we are all done and can complete.
  for ( port_t const& port : complete_ports )
  {
    d->group_data.erase( port );
  }

  if ( d->group_data.empty() )
  {
    LOG_TRACE( logger(), "Process complete" );
    mark_process_as_complete();
  }
} // mux_process::_step


// ------------------------------------------------------------------
process::properties_t
mux_process
::_properties() const
{
  properties_t consts = process::_properties();

  consts.insert( property_unsync_input );

  return consts;
}


// ------------------------------------------------------------------
// Intercept input port connection so we can create the requested port
void
mux_process
::input_port_undefined( port_t const& port )
{
  // Accepts connections from "<group>/<item>" and create
  // output "res/<group>" output port the first time.

  LOG_TRACE( logger(), "Processing input port: \"" << port << "\"" );

  // Extract GROUP sub-string from port name
  ports_t components;
  kwiver::vital::tokenize( port, components, priv::res_sep );

  // Results are:
  // components[0] = group
  // components[1] = item

  // Port name must start with "in"
  if ( components.size() == 2 )
  {
    const priv::group_t group = components[0];

    // If GROUP does not exist, then this is the first port in the group
    if ( 0 == d->group_data.count( group ) )
    {
      // This is the first port of this group. Need to make output port.
      priv::group_info info;
      d->group_data[group] = info;

      port_flags_t required;
      required.insert( flag_required );

      LOG_TRACE( logger(), "Creating output port: \"" << group << "\"" );

      // Create output port "res/group"
      declare_output_port(
        group,
        type_flow_dependent + group, // note the group magic on port type
        required,
        port_description_t( "The output port for " + group + "." ) );
    }

    // Get entry based on the group string
    priv::group_info& info = d->group_data[group];

    // If this "item" is not already in the port list, then add it.
    if ( std::find( info.ports.begin(), info.ports.end(), port ) == info.ports.end() )
    {
      // Add this port to the info list for this group
      info.ports.push_back( port );

      port_flags_t required;
      required.insert( flag_required );

      LOG_TRACE( logger(), "Creating input port: \"" << port << "\"" );

      // Open an input port for the name
      declare_input_port(
        port,
        type_flow_dependent + group, // note the group magic on port type
        required,
        port_description_t( "An input for the " + group + " data." ) );
    }
  }
  else
  {
    // This may not be the best way to handle this type of
    // error. Maybe an exception would be better. In any event, the
    // connection will fail because the port will not be defined.
    LOG_ERROR( logger(), "Input port \"" << port << "\" does not have the correct format. "
                                                   "Must be in the form \"<group>/<item>\"." );
  }

} // mux_process::_input_port_info


// ------------------------------------------------------------------
mux_process::priv
::priv()
  : group_data(),
  m_config_term_policy( skip )
{
}


mux_process::priv
::~priv()
{
}


// ------------------------------------------------------------------
mux_process::priv::group_info
::group_info()
  : ports(),
    cur_port()
{
}


mux_process::priv::group_info
::~group_info()
{
}


} // end namespace
