#ifndef KWIVER_VITAL_PYTHON_TRAIN_DETECTOR_H_
#define KWIVER_VITAL_PYTHON_TRAIN_DETECTOR_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace kwiver {
namespace vital {
namespace python {

void train_detector( py::module &m );

}
}
}
#endif
