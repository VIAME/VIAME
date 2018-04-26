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

#ifndef _KWIVER_INITIALIZE_OBJECT_TRACKS_PROCESS_H_
#define _KWIVER_INITIALIZE_OBJECT_TRACKS_PROCESS_H_

#include "kwiver_processes_export.h"

#include <sprokit/pipeline/process.h>

#include <memory>

namespace kwiver
{

// -----------------------------------------------------------------------------
/**
 * \class initialize_object_tracks_process
 *
 * \brief Initialized new tracks given object detections.
 *
 *  On optional input track port will union the input track set and newly
 *  initialized tracks.
 * 
 * \iports
 * \iport{timestamp}
 * \iport{image}
 * \iport{detections}
 * \iport{tracks}
 *
 * \oports
 * \oport{tracks}
 */
class KWIVER_PROCESSES_NO_EXPORT initialize_object_tracks_process
  : public sprokit::process
{
  public:
  initialize_object_tracks_process( vital::config_block_sptr const& config );
  virtual ~initialize_object_tracks_process();

  protected:
    virtual void _configure();
    virtual void _step();
    virtual void _init();

  private:
    void make_ports();
    void make_config();

    class priv;
    const std::unique_ptr<priv> d;
 }; // end class initialize_object_tracks_process


} // end namespace
#endif /* _KWIVER_INITIALIZE_OBJECT_TRACKS_PROCESS_H_ */
