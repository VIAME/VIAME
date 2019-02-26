#ifndef PY_IMAGE_OBJECT_DETECTOR_TCC
#define PY_IMAGE_OBJECT_DETECTOR_TCC


#include <pybind11/pybind11.h>
#include <vital/algo/image_object_detector.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/image_container.h>
#include "algorithm_trampoline.tcc"

using namespace kwiver::vital::algo;
using namespace kwiver::vital;


template <class algorithm_def_iod_base=algorithm_def<image_object_detector>>
class py_iod_algorithm_def 
                          : public py_algorithm<algorithm_def_iod_base>
{
  public:
    using py_algorithm<algorithm_def_iod_base>::py_algorithm;

    std::string type_name() const override 
    {
      PYBIND11_OVERLOAD(
        std::string,
        algorithm_def<image_object_detector>,
        type_name,
      );
    }
};


template <class image_object_detector_base=image_object_detector>
class py_image_object_detector 
                : public py_iod_algorithm_def<image_object_detector_base>
{
  public:
    using py_iod_algorithm_def<image_object_detector_base>::py_iod_algorithm_def;
        
    detected_object_set_sptr detect(image_container_sptr image_data) const override 
    {                  
      PYBIND11_OVERLOAD_PURE(                                                   
        detected_object_set_sptr,                                                            
        image_object_detector,                                                        
        detect,                                                                 
        image_data 
      );                                                             
    }           
};

#endif
