// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/extract_descriptors_extras.h>

#include <viame/algorithm_framework/algo/extract_descriptors.h>

#include <memory>
#include <utility>

namespace kwiver::vital::python {

namespace py = pybind11;

void
extract_descriptors_extras( py::module& m )
{
  // The generated binding exposes extract() exactly as the C++ signature
  // reads, but the C++ signature takes the feature set by reference and may
  // replace it: an extractor is allowed to reorder, drop, or duplicate
  // features so they align with the descriptors it returns (the OCV one always
  // rebuilds the set). A python caller handed only the descriptor set is left
  // holding a feature set that no longer corresponds to it, with no way to
  // notice. Replace the generated method with one returning both, which is the
  // contract python callers such as viame's stabilize_many_images were written
  // against.
  py::object cls = m.attr( "ExtractDescriptors" );
  cls.attr( "extract" ) = py::cpp_function(
    []( kwiver::vital::algo::extract_descriptors const& self,
        kwiver::vital::image_container_sptr image_data,
        kwiver::vital::feature_set_sptr features,
        kwiver::vital::image_container_sptr image_mask )
    {
      auto descriptors = self.extract( image_data, features, image_mask );
      // features may have been replaced, so return it alongside
      return std::make_pair( std::move( descriptors ), std::move( features ) );
    },
    py::is_method( cls ),
    py::doc(
      "Extract descriptors for each feature. Returns (descriptors, features); "
      "the returned feature set may differ from the input, reordered or "
      "reduced to match the descriptors." ),
    py::arg( "image_data" ),
    py::arg( "features" ),
    py::arg( "image_mask" ) = py::none() );
}

} // namespace kwiver::vital::python
