#ifndef PY_ALGORITHM_TRAMPOLINE_TCC
#define PY_ALGORITHM_TRAMPOLINE_TCC

#include <pybind11/pybind11.h>
#include <vital/algo/algorithm.h>
#include <vital/config/config_block.h>


template <class algorithm_base=kwiver::vital::algorithm> 
class py_algorithm : public algorithm_base 
{
  public:
    using algorithm_base::algorithm_base;

    std::string type_name() const override 
    {
      PYBIND11_OVERLOAD_PURE(
        std::string,
        algorithm_base,
        type_name,      
      );
    }

    kwiver::vital::config_block_sptr get_configuration() const override 
    {
      PYBIND11_OVERLOAD(
        kwiver::vital::config_block_sptr,
        algorithm_base,
        get_configuration,      
      );
    }

    void set_configuration(kwiver::vital::config_block_sptr config) override 
    {
      PYBIND11_OVERLOAD_PURE(
        void,
        algorithm_base,
        set_configuration,
        config      
      );
    }
    
    bool check_configuration(kwiver::vital::config_block_sptr config) const override 
    {
      PYBIND11_OVERLOAD_PURE(
        bool,
        algorithm_base,
        check_configuration,
        config      
      );
    }
};

#endif
