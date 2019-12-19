#ifndef KWIVER_VITAL_PYTHON_ASSOCIATE_DETECTIONS_TO_TRACKS_H_
#define KWIVER_VITAL_PYTHON_ASSOCIATE_DETECTIONS_TO_TRACKS_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void associate_detections_to_tracks(py::module &m);
#endif
