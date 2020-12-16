// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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