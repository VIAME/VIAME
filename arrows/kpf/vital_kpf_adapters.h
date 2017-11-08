/*ckwg +29
* Copyright 2016-2017 by Kitware, Inc.
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
* \brief adapters for writing kpf to and from vital objects
*/

namespace KPF = kwiver::vital::kpf;
namespace KPFC = KPF::canonical;//
                                // This is the socially-agreed domain for the detector; an arbitrary number
                                // greater than 9 which disambiguates this detector from others we may
                                // have on the project.
                                //
const int DETECTOR_DOMAIN = 17;

//
// For non-scalar data which is represented by a non-scalar KPF structure
// (e.g. a bounding box), you need to define two routines: one converts
// your structure into KPF, the other converts KPF into your structure.
//
// Note that since the box is "distributed" across several fields of
// user_complex_detection_t, the adapter needs to take an instance of the
// complete user type, even though it's only setting a few fields.

//
// The KPF 'canonical' bounding box is (x1,y1)-(x2,y2).
//

struct vital_box_adapter_t : public KPF::kpf_box_adapter< kwiver::vital::bounding_box_d >
{
  vital_box_adapter_t() :
    kpf_box_adapter< kwiver::vital::bounding_box_d >(
      // reads the canonical box into the bounding_box_d
      [](const KPF::canonical::bbox_t& b, kwiver::vital::bounding_box_d& bbox) {
    kwiver::vital::bounding_box_d tmp(b.x1, b.y1, b.x2, b.y2);
    bbox = tmp; },

      // converts a bounding_box_d into a canonical box and returns it
      [](const kwiver::vital::bounding_box_d& bbox) {
      return KPF::canonical::bbox_t(
        bbox.min_x(),
        bbox.min_y(),
        bbox.max_x(),
        bbox.max_y()); })
  {}
};

/* I do not think vital has this data associated with detected objects...
struct vital_poly_adapter_t : public KPF::kpf_poly_adapter< kwiver::vital::detected_object >
{
  vital_poly_adapter_t() :
    kpf_poly_adapter< kwiver::vital::detected_object >(
      // reads the canonical box "b" into the user_detection "d"
      [](const KPF::canonical::poly_t& b, kwiver::vital::detected_object& d) {
    d.poly_x.clear(); d.poly_y.clear();
    for (auto p : b.xy) {
      d.poly_x.push_back(p.first);
      d.poly_y.push_back(p.second);
    }},
      // converts a user_detection "d" into a canonical box and returns it
      [](const kwiver::vital::detected_object& d) {
      KPF::canonical::poly_t p;
      // should check that d's vectors are the same length
      for (size_t i = 0; i<d.poly_x.size(); ++i) {
        p.xy.push_back(make_pair(d.poly_x[i], d.poly_y[i]));
      }
      return p; })
  {}
};
*/