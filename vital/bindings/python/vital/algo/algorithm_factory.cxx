#include <vital/algo/algorithm.h>
#include <vital/algo/algorithm_factory.h>
#include <pybind11/pybind11.h>
#include <iostream>
namespace py = pybind11;

static void add_algorithm( const std::string& impl_name, std::string const& description, 
                            py::object conc_t );
void mark_algorithm_as_loaded( const std::string& module_name );


typedef std::function< py::object() > py_algorithm_factory_func_t;


class python_algorithm_factory : public kwiver::vital::algorithm_factory
{
  public:
    python_algorithm_factory( const std::string& algo,
                              const std::string& impl,
                              py_algorithm_factory_func_t conc_f )
      : kwiver::vital::algorithm_factory( algo, impl ),
        m_conc_f(conc_f)
      
    {
      this->add_attribute( CONCRETE_TYPE, impl );
    }

    virtual ~python_algorithm_factory()=default;
    
  protected:
    kwiver::vital::algorithm_sptr create_object_a();

  private:
    py_algorithm_factory_func_t m_conc_f;
};

kwiver::vital::algorithm_sptr python_algorithm_factory::create_object_a()
{
  py::object obj = m_conc_f();
  // Necessary or pybind11 creates an object of ImageObjectDetector
  obj.inc_ref();
  kwiver::vital::algorithm_sptr algo_sptr = obj.cast<kwiver::vital::algorithm_sptr>();
  return algo_sptr;
}


PYBIND11_MODULE(algorithm_factory, m)
{
  m.def("has_algorithm_impl_name", &kwiver::vital::has_algorithm_impl_name, 
        py::arg("type_name"), py::arg("impl_name"),
        "Returns True if the algorithm implementation has been registered");
  m.def("add_algorithm", &add_algorithm, "Registers an algorithm");
  m.def("mark_algorithm_as_loaded", &mark_algorithm_as_loaded, 
      "Marks the algorithm as loaded");
}


static void add_algorithm( std::string const& impl_name, std::string const& description, 
                    py::object conc_f)
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  std::string type_name = py::str( conc_f.attr("static_type_name")() );
  
  //python_algorithm_wrapper const& wrap(conc_t);
  auto fact  = vpm.add_factory( new python_algorithm_factory( type_name, 
                                                              impl_name,
                                                              conc_f ));
  
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, impl_name)
      .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, "python-runtime" )
      .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, description );  
}

void mark_algorithm_as_loaded( const std::string& name )
{
    kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
    vpm.mark_module_as_loaded( name );
}
 
