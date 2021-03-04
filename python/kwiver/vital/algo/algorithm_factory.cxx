// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file algorithm_factory.cxx
 *
 * \brief Python bindings for \link vital::algorithm_factory\endlink
 */

#include <vital/algo/algorithm.h>
#include <vital/algo/algorithm_factory.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <iostream>

#include <python/kwiver/vital/util/pybind11.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
static void add_algorithm( const std::string& impl_name, std::string const& description,
                            py::object conc_t );

void mark_algorithm_as_loaded( const std::string& module_name );
static std::vector< std::string > implementation_names(const std::string& algorithm_name);

class python_algorithm_factory : public kwiver::vital::algorithm_factory
{
  public:
    python_algorithm_factory( const std::string& algo,
                              const std::string& impl,
                              py::object conc_f );

    virtual ~python_algorithm_factory()=default;
  protected:
    kwiver::vital::algorithm_sptr create_object_a();

  private:
    py::object m_conc_f;
};
}
}
}
using namespace kwiver::vital::python;

PYBIND11_MODULE(algorithm_factory, m)
{
  m.def("has_algorithm_impl_name", &kwiver::vital::has_algorithm_impl_name,
        py::call_guard<kwiver::vital::python::gil_scoped_release>(),
        py::arg("type_name"), py::arg("impl_name"),
        "Returns True if the algorithm implementation has been registered");

  m.def("add_algorithm", &add_algorithm,
      py::call_guard<kwiver::vital::python::gil_scoped_release>(),
      "Registers an algorithm");

  m.def("mark_algorithm_as_loaded", &mark_algorithm_as_loaded,
      py::call_guard<kwiver::vital::python::gil_scoped_release>(),
      "Marks the algorithm as loaded");

  m.def("implementations", &implementation_names,
      py::call_guard<kwiver::vital::python::gil_scoped_release>(),
      "Returns all the implementations of an algorithm");

  m.def("create_algorithm", &kwiver::vital::create_algorithm,
      py::call_guard<kwiver::vital::python::gil_scoped_release>(),
      py::arg("algo_name"), py::arg("impl_name"),
      "Creates a concrete instance of algo specified by impl_name");
}

namespace kwiver {
namespace vital  {
namespace python {
python_algorithm_factory::python_algorithm_factory( const std::string& algo,
                          const std::string& impl,
                          py::object conc_f )
  : kwiver::vital::algorithm_factory( algo, impl ),
    m_conc_f(conc_f)
{
  this->add_attribute( CONCRETE_TYPE, impl );
}

kwiver::vital::algorithm_sptr python_algorithm_factory::create_object_a()
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  py::object obj = m_conc_f();
  obj.inc_ref();
  kwiver::vital::algorithm_sptr algo_sptr = obj.cast<kwiver::vital::algorithm_sptr>();
  return algo_sptr;
}

static void add_algorithm( std::string const& impl_name, std::string const& description,
                            py::object conc_f)
{
  using kvpf = kwiver::vital::plugin_factory;

  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  std::string type_name = py::str( conc_f.attr("static_type_name")() );
  auto fact  = vpm.add_factory( new python_algorithm_factory( type_name,
                                                              impl_name,
                                                              conc_f ));

  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, impl_name)
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, description )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_CATEGORY, kvpf::ALGORITHM_CATEGORY )
    ;
}

void mark_algorithm_as_loaded( const std::string& name )
{
    kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
    vpm.mark_module_as_loaded( name );
}

// ------------------------------------------------------------------
bool is_algorithm_loaded( const std::string& name )
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  return vpm.is_module_loaded( name );
}

static std::vector< std::string > implementation_names(const std::string& algorithm_name)
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  auto fact_list = vpm.get_factories(algorithm_name);

  std::vector< std::string > all_implementations;
  for(auto fact : fact_list )
  {
    std::string buf;
    if ( fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, buf ) )
    {
      all_implementations.push_back( buf );
    }
  }

  return all_implementations;
}
}}}
