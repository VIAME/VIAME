#ifndef PY_IMAGE_OBJECT_DETECTOR
#define PY_IMAGE_OBJECT_DETECTOR

#include <pybind11/pybind11.h>
#include <vital/algo/image_object_detector.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/image_container.h>
#include "algorithm_trampoline.cxx"

using namespace kwiver::vital::algo;
using namespace kwiver::vital;


class py_image_object_detector_algorithm_def : public algorithm_def<image_object_detector>
{
  public:
    using algorithm_def<image_object_detector>::algorithm_def;

    std::string type_name() const override 
    {
      PYBIND11_OVERLOAD(
        std::string,
        algorithm_def<image_object_detector>,
        type_name,
      );
    }

    config_block_sptr get_configuration() const override 
    {
      PYBIND11_OVERLOAD_PURE(
        config_block_sptr,
        algorithm_def<image_object_detector>,
        get_configuration,      
      );
    }

    void set_configuration(config_block_sptr config) override 
    {
      PYBIND11_OVERLOAD_PURE(
        void,
        algorithm_def<image_object_detector>,
        set_configuration,
        config      
      );
    }
    
    bool check_configuration(config_block_sptr config) const override 
    {
      PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algorithm_def<image_object_detector>,
        check_configuration,
        config      
      );
    }

};

class py_image_object_detector : public image_object_detector
{
  public:
    using image_object_detector::image_object_detector;

    std::string type_name() const override 
    {                                    
      PYBIND11_OVERLOAD(                                                        
         std::string,                                                            
         image_object_detector,                                                        
         type_name,                                                              
      );                                                                      
    }                                                                           
        
    detected_object_set_sptr detect(image_container_sptr image_data) const override 
    {                  
      PYBIND11_OVERLOAD_PURE(                                                   
        detected_object_set_sptr,                                                            
        image_object_detector,                                                        
        detect,                                                                 
        image_data 
      );                                                             
    }           
    
    config_block_sptr get_configuration() const override 
    {
      PYBIND11_OVERLOAD_PURE(
        config_block_sptr,
        image_object_detector,
        get_configuration,      
      );
    }

    void set_configuration(config_block_sptr config) override 
    {
      PYBIND11_OVERLOAD_PURE(
        void,
        image_object_detector,
        set_configuration,
        config      
      );
    }
    
    bool check_configuration(config_block_sptr config) const override 
    {
      PYBIND11_OVERLOAD_PURE(
        bool,
        image_object_detector,
        check_configuration,
        config      
      );
    }
};

#endif
