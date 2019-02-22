#ifndef PY_ALGORITHM_TRAMPOLINE_H
#define PY_ALGORITHM_TRAMPOLINE_H

#include <pybind11/pybind11.h>
#include <vital/algo/algorithm.h>
#include <vital/config/config_block.h>

using namespace kwiver::vital;

class py_algorithm : public algorithm 
{
  public:
    using algorithm::algorithm;

    std::string type_name() const override 
    {
      PYBIND11_OVERLOAD_PURE(
        std::string,
        kwiver::vital::algorithm,
        type_name,      
      );
    }

    config_block_sptr get_configuration() const override 
    {
      PYBIND11_OVERLOAD_PURE(
        config_block_sptr,
        kwiver::vital::algorithm,
        get_configuration,      
      );
    }

    void set_configuration(config_block_sptr config) override 
    {
      PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algorithm,
        set_configuration,
        config      
      );
    }
    
    bool check_configuration(config_block_sptr config) const override 
    {
      PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algorithm,
        check_configuration,
        config      
      );
    }
};
#endif
