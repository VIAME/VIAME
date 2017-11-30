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
 * \brief Implementation for detected_object_set_input_kpf
 */

#include "detected_object_set_input_kpf.h"

#include "yaml/kpf_reader.h"
#include "yaml/kpf_yaml_parser.h"
#include "yaml/kpf_canonical_io_adapter.h"

#include "vital_kpf_adapters.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>
#include <vital/logger/logger.h>
#include <vital/exceptions.h>

#include <map>
#include <sstream>
#include <cstdlib>

namespace kwiver {
namespace arrows {
namespace kpf {

// ------------------------------------------------------------------
class detected_object_set_input_kpf::priv
{
public:
  priv( detected_object_set_input_kpf* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "detected_object_set_input_kpf" ) )
    , m_first( true )
  { }

  ~priv() { }

  void read_all();

  detected_object_set_input_kpf* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;

  int m_current_idx;
  int m_last_idx;

  // Map of detected objects indexed by frame number. Each set
  // contains all detections for a single frame.
  std::map< int, kwiver::vital::detected_object_set_sptr > m_detected_sets;
};


// ==================================================================
detected_object_set_input_kpf::
detected_object_set_input_kpf()
  : d( new detected_object_set_input_kpf::priv( this ) )
{
}


detected_object_set_input_kpf::
~detected_object_set_input_kpf()
{
}


// ------------------------------------------------------------------
void
detected_object_set_input_kpf::
set_configuration(vital::config_block_sptr config)
{ }


// ------------------------------------------------------------------
bool
detected_object_set_input_kpf::
check_configuration(vital::config_block_sptr config) const
{
  return true;
}


// ------------------------------------------------------------------
bool
detected_object_set_input_kpf::
read_set( kwiver::vital::detected_object_set_sptr & set, std::string& image_name )
{
  if ( d->m_first )
  {
    // Read in all detections
    d->read_all();
    d->m_first = false;

    // set up iterators for returning sets.
    d->m_current_idx = d->m_detected_sets.begin()->first;
    d->m_last_idx = d->m_detected_sets.rbegin()->first;
  } // end first

  // we do not return image name
  image_name.clear();

  // return detection set at current index if there is one
  if ( 0 == d->m_detected_sets.count( d->m_current_idx ) )
  {
    // return empty set
    set = std::make_shared< kwiver::vital::detected_object_set>();
  }
  else
  {
    // Return detections for this frame.
    set = d->m_detected_sets[d->m_current_idx];
  }

  ++d->m_current_idx;

  return true;
}


// ------------------------------------------------------------------
void
detected_object_set_input_kpf::
new_stream()
{
  d->m_first = true;
}


// ==================================================================
void
detected_object_set_input_kpf::priv::
read_all()
{
  m_detected_sets.clear();

  

  KPF::kpf_yaml_parser_t parser(m_parent->stream());
  KPF::kpf_reader_t reader(parser);

  // Is there a way to confirm this is a kpf of detections only format?
  //if (? ? )
  //{
  //  std::stringstream str;
  //  str << "This is not a kpf file; found " << col.size()
  //    << " columns in\n\"" << line << "\"";
  //  throw kwiver::vital::invalid_data(str.str());
  //}

  size_t      detection_id;
  double      frame_number;
  std::string detector_name;
  double      confidence;
  vital_box_adapter_t box_adapter;
  kwiver::vital::detected_object_type_sptr types(new kwiver::vital::detected_object_type());
  kwiver::vital::detected_object_set_sptr frame_detections;

  while (reader
    >> KPF::reader< KPFC::kv_t>("detector_name", detector_name)
    >> KPF::reader< KPFC::id_t >(detection_id, KPFC::id_t::DETECTION_ID)
    >> KPF::reader< KPFC::timestamp_t>(frame_number, KPFC::timestamp_t::FRAME_NUMBER)
    >> KPF::reader< KPFC::conf_t>(confidence, DETECTOR_DOMAIN)
    >> KPF::reader< KPFC::bbox_t >(box_adapter, KPFC::bbox_t::IMAGE_COORDS)
    )
  {
    kwiver::vital::bounding_box_d bbox(0, 0, 0, 0);
    box_adapter.get(bbox);
    kwiver::vital::detected_object_sptr det(new kwiver::vital::detected_object(bbox, confidence, types));
    det->set_detector_name(detector_name);
    
    frame_detections = m_detected_sets[frame_number];
    if (frame_detections.get() == nullptr)
    {
      // create a new detection set entry
      frame_detections = std::make_shared<kwiver::vital::detected_object_set>();
      m_detected_sets[frame_number] = frame_detections;
    }
    frame_detections->add(det);

    // did we receive any metadata?
    for (auto m : reader.get_meta_packets())
    {
      std::cout << "Metadata: '" << m << "'\n";
    }

    reader.flush();
  }

} // read_all

} } } // end namespace
