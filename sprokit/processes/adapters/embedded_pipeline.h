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
 * \brief Interface file for the embedded pipeline.
 */

#ifndef ARROWS_PROCESSES_EMBEDDED_PIPELINE_H
#define ARROWS_PROCESSES_EMBEDDED_PIPELINE_H

#include <sprokit/processes/adapters/kwiver_adapter_export.h>

#include "adapter_data_set.h"

#include <vital/logger/logger.h>

#include <istream>
#include <string>

namespace kwiver {

// -----------------------------------------------------------------
/**
 * @brief Embedded callable pipeline.
 *
 * This class implements a sprokit pipeline that can be instantiated
 * within a program.
 *
 * Inputs to the pipeline are passed to the input adapter through the
 * send() method. Outputs from the pipeline are retrieved using the
 * receive() method.
 *
 * The pipeline description must contain no more than one input
 * adapter (process type "input_adapter") and no more than one output
 * adapter (process type "output_adapter"). The actual process names
 * are up to you.  The pipeline should be configured so that the
 * inputs to the pipeline come from the input_adapter and the outputs
 * go to the output_adapter. There are no other constraints on the
 * pipeline topology.
 *
 * Refer to the documentation for generating pipelines that contain
 * source or sink processes.
 *
 * When creating a data set for the input adapter, there must be a
 * datum for each port on the input_adapter_process. The process will
 * throw an exception if there is a datum for a port that is not
 * connected or there is a port that does not have a datum in the set.
 *
 * Example:
\code
  #include <sprokit/pipeline_util/literal_pipeline.h>

  // SPROKIT macros can be used to create pipeline description
  std::stringstream pipeline_desc;
  pipeline_desc << SPROKIT_PROCESS( "input_adapter",  "ia" )
                << SPROKIT_PROCESS( "output_adapter", "oa" )

                << SPROKIT_CONNECT( "ia", "counter",  "oa", "out_num" )
                << SPROKIT_CONNECT( "ia", "port2",    "oa", "port3" )
                << SPROKIT_CONNECT( "ia", "port3",    "oa", "port2" )
    ;

  // create embedded pipeline
  kwiver::embedded_pipeline ep;
  ep.build_pipeline( pipeline_desc );

  // Query adapters for ports
  auto input_list = ep.input_port_names();
  auto output_list = ep.output_port_names();

  // Verify ports are as expected. (if needed)
  // This is more likely needed if the pipeline description is read from
  // an external file where external forces can affect the validity of
  // the pipeline description.
  // ...

  // Start pipeline
  ep.start();

  for ( int i = 0; i < 10; ++i)
  {
    // Create dataset for input
    auto ds = kwiver::adapter::adapter_data_set::create();

    // Add value to be pushed to the named port
    ds.add_value( "counter", i );

    // Data values need to be supplied to all connected ports
    // (based on the previous pipeline definition)
    ds.add_value( "port2", i );
    ds.add_value( "port3", i );
    ep.send( ds ); // push into pipeline

    // Get output from pipeline
    auto rds = ep.receive();

    // get value from the output adapter
    int val = rds->get_port_data<int>( "out_num" );

    // val should be the same as i

  } // end for

  ep.send_end_of_input(); // indicate end of input

  auto rds = ep.receive(); // Retrieve end of input data item.
  if ( ! ep.at_end() || ! rds.is_end_of_data() )
  {
    // This is unexpected.
  }
\endcode
 */
class KWIVER_ADAPTER_EXPORT embedded_pipeline
{
public:
  /**
   * @brief Create embedded pipeline from description in stream.
   *
   */
  embedded_pipeline();
  virtual ~embedded_pipeline();

  /**
   * @brief Build the embedded pipeline.
   *
   * This method creates the pipeline based on the contents of the
   * supplied stream.
   *
   * @param istr Input stream containing the pipeline description.
   *
   * @param def_dir The directory name used to report errors in the
   * input stream and is used as the current directory to locate
   * includes and to resolve relpath. Since the input stream being
   * processed has no file name, the name "in-stream" is appended to
   * the directory supplied so that errors in the stream can be
   * differentiated from errors from other files. If this parameter is
   * not supplied, the current directory is used.
   *
   * @throws std::runtime_error when there is a problem
   * constructing the pipeline or if there is a problem connecting
   * inputs or outputs.
   */
  void build_pipeline( std::istream& istr, std::string const& def_dir = "" );

  /**
   * @brief Send data set to input adapter.
   *
   * This method sends a data set object to the input adapter. The
   * adapter data set must contain a datum for each port on the input
   * adapter process.
   *
   * If the pipeline is full and can not accept the data set, this
   * method will block until the pipeline can accept the input.
   *
   * The end-of-data item is sent to the pipeline after the last data
   * item to indicate that there are no more data ant the pipeline
   * should start an orderly termination. Passing more data after the
   * end-of-data set has been sent is not a good idea.
   *
   * @param ads Data set to send
   */
  void send( kwiver::adapter::adapter_data_set_t ads );

  /**
   * @brief Send end of input into pipeline.
   *
   * This method indicates that there will be no more input into the
   * pipeline. The pipeline starts to shutdown after this method is
   * called. Calling the wait() method will block until the pipeline
   * processing is complete.
   *
   * Calling send() after this method is called is not a good
   * idea.
   */
  void send_end_of_input();

  /**
   * @brief Get pipeline output data.
   *
   * This method returns a data set produced by the pipeline. It will
   * contain one entry for each port on the output adapter process.
   *
   * If the is no output data set immediately available, this call
   * will block until one is available.
   *
   * The last data set returned from the pipeline will be marked as
   * end of data (is_end_of_data() returns true). After this end of
   * data marker has been processed by this receive() method, the
   * embedded pipeline is marked as being at the end of data. This
   * status can be checked by calling the at_end() method.
   *
   * Calling this receive() method after the end of data item has been
   * returned is not a good idea as it can cause a deadlock.
   *
   * @return Data set from the pipeline.
   */
  kwiver::adapter::adapter_data_set_t receive();

  /**
   * @brief Can pipeline accept more input?
   *
   * This method checks to see if the input adapter process can accept
   * more data. This method can be used to create a polling
   * (non-blocking) approach to sending data to the pipeline.
   *
   * @return \b true if interface queue is full.
   */
  bool full() const;

  /**
   * @brief Is any pipeline output ready?
   *
   * This method checks to see if there is a pipeline output data set
   * ready. This method can be used if the pipeline owner is polling
   * for output.
   *
   * @return \b true if interface queue is empty.
   */
  bool empty() const;

  /**
   * @brief Is pipeline terminated.
   *
   * This method returns true if the end of input marker has been
   * received from the pipeline, indicating that the pipeline has
   * processed all the data and terminated.
   *
   * @return \b true if all data has been processed and pipeline has terminated.
   */
  bool at_end() const;

  /**
   * @brief Start the pipeline
   *
   * This method starts the pipeline processing. After this call, the
   * pipeline is ready to accept input data sets.
   *
   * Calling start() on a pipeline that is already started results in
   * undefined behaviour.
   */
  void start();

  /**
   * @brief Wait for pipeline to complete.
   *
   * This method waits until the pipeline scheduler terminates. This
   * is useful when terminating an embedded pipeline to make sure that
   * all threads have terminated.
   *
   * Calling this \b before sending an end-of-input has been sent to
   * the pipeline will block the caller until the pipeline terminates,
   * most likely causing a deadlock.
   */
  void wait();

  /**
   * @brief Stop an executing pipeline.
   *
   * This method signals the pipeline to stop and waits until it has
   * terminated.
   */
  void stop();

  /**
   * @brief Get list of input ports.
   *
   * This method returns the list of all active data ports on the
   * input adapter. This list can used to drive the adapter_data_set
   * creation so that here is a datum of the correct type for each
   * port.
   *
   * The actual port names are specified in the pipeline
   * configuration.
   *
   * @return List of input port names
   */
  sprokit::process::ports_t input_port_names() const;

  /**
   * @brief Get list of output ports.
   *
   * This method returns the list of all active data ports on the
   * output adapter. This list can used to process the
   * adapter_data_set returned by the receive() method.  There will be
   * a datum for each output port in the returned data set.
   *
   * The actual port names are specified in the pipeline
   * configuration.
   *
   * @return List of output port names
   */
  sprokit::process::ports_t output_port_names() const;

  /**
   * @brief Report if input adapter is connected.
   *
   * This method determines if the input adapter is connected.
   *
   * @return \b true if input adapter is connected.
   */
  bool input_adapter_connected() const;

  /**
   * @brief Report if output adapter is connected.
   *
   * This method determines if the output adapter is connected.
   *
   * @return \b true if output adapter is connected.
   */
  bool output_adapter_connected() const;

  class priv;

protected:
  /**
   * @brief Connect to input adapter.
   *
   * This method connects the external data path to the input
   * process. Derived classes can override this method if different
   * input handling is needed, such as letting the pipeline supply the
   * data source process.
   */
  virtual bool connect_input_adapter();

  /**
   * @brief Connect to output adapter.
   *
   * This method connects the external data path to the output
   * process. Derived classes can override this method if different
   * output handling is needed, such as letting the pipeline supply
   * the data sink process.
   */
  virtual bool connect_output_adapter();

  /**
   * @brief Update pipeline config.
   *
   * This method provides the ability for a derived class to inspect
   * and update the pipeline config before it is used to create the
   * pipeline. Additional config entries can be added or existing ones
   * modified to suit a specific application.
   *
   * The default implementation does not modify the config in any way.
   *
   * @param[in,out] config Configuration to update.
   */
  virtual void update_config( kwiver::vital::config_block_sptr config );

private:
  std::shared_ptr< priv > m_priv;

}; // end class embedded_pipeline

} // end namespace

#endif /* ARROWS_PROCESSES_EMBEDDED_PIPELINE_H */
