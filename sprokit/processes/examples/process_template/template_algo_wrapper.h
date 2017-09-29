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
//++ The above header applies to this template code. Feel free to use your own
//++ license header.

/**
 * \file
 * \brief Interface to template process.
 */

//++ Change name in include guard to match the process name.
//++ Use a similar style for all processes to make sure there is no collision.
#ifndef GROUP_TEMPLATE_ALGO_WRAPPER_H
#define GROUP_TEMPLATE_ALGO_WRAPPER_H

#include <sprokit/pipeline/process.h>

//++ Include export header for the plugin the process belongs to.
#include "template_processes_export.h"

#include <memory>

namespace group_ns {

// ----------------------------------------------------------------
/**
 * @brief brief description
 *
 */
//++ Processes are not exported, so add the appropriate NO_EXPORT
//++ symbol to the class.
class TEMPLATE_PROCESSES_NO_EXPORT template_algo_wrapper
  : public sprokit::process
{
public:
  template_algo_wrapper( kwiver::vital::config_block_sptr const& config );
  virtual ~template_algo_wrapper();

protected:
  //++ This is a list of all methods that a process can supply.
  //++ configure() and step() are generally needed.  Not all of the
  //++ others are needed in every process, so they can be omitted if not
  //++ used.
  virtual void _configure();
  virtual void _step();

  //++ These methods are not usually needed by a simple process.
  //++ They are only included for completeness. If not needed then delete them.
  virtual void _init();
  virtual void _reset();
  virtual void _flush();
  virtual void _reconfigure(kwiver::vital::config_block_sptr const& conf);

private:
  //++ these methods group config creation operations and port
  //++ creation operations so they are easy to locate.
  void make_ports();
  void make_config();

  //++ Processes generally use a private implementation idiom (pimpl)
  class priv;
  const std::unique_ptr<priv> d;

}; // end class

}

#endif // GROUP_TEMPLATE_ALGO_WRAPPER_H
