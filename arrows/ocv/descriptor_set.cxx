// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV descriptor_set implementation
 */

#include "descriptor_set.h"

#include <vital/exceptions.h>

/// This macro applies another macro to all of the types listed below.
#define APPLY_TO_TYPES(MACRO) \
    MACRO(kwiver::vital::byte); \
    MACRO(float); \
    MACRO(double)

namespace kwiver {
namespace arrows {
namespace ocv {

namespace
{

/// Templated helper function to convert matrix row into a descriptor
template <typename T>
vital::descriptor_sptr
ocv_to_vital_descriptor(const cv::Mat& v)
{
  vital::descriptor_array_of<T>* d = NULL;
  switch(v.cols)
  {
  case 64:
    d = new vital::descriptor_fixed<T,64>();
    break;
  case 128:
    d = new vital::descriptor_fixed<T,128>();
    break;
  case 256:
    d = new vital::descriptor_fixed<T,256>();
    break;
  default:
    d = new vital::descriptor_dynamic<T>(v.cols);
  }
  std::copy(v.begin<T>(), v.end<T>(), d->raw_data());
  return vital::descriptor_sptr(d);
}

/// Templated helper function to convert descriptors into a cv::Mat
template <typename T>
cv::Mat
vital_descriptors_to_ocv(const std::vector<vital::descriptor_sptr>& desc)
{
  const unsigned int num = static_cast<unsigned int>(desc.size());
  const unsigned int dim = static_cast<unsigned int>(desc[0]->size());
  cv::Mat_<T> mat(num,dim);
  for( unsigned int i=0; i<num; ++i )
  {
    const vital::descriptor_array_of<T>* d =
        dynamic_cast<const vital::descriptor_array_of<T>*>(desc[i].get());
    if( !d || d->size() != dim )
    {
      VITAL_THROW( vital::invalid_value, "mismatch type or size when converting descriptors to OpenCV");
    }
    cv::Mat_<T> row = mat.row(i);
    std::copy(d->raw_data(), d->raw_data() + dim, row.begin());
  }
  return mat;
}

/// Convert OpenCV type number into a string
std::string cv_type_to_string(int number)
{
    // find type
    int type_int = number % 8;
    std::string type_str;

    switch (type_int)
    {
        case 0:
            type_str = "8U";
            break;
        case 1:
            type_str = "8S";
            break;
        case 2:
            type_str = "16U";
            break;
        case 3:
            type_str = "16S";
            break;
        case 4:
            type_str = "32S";
            break;
        case 5:
            type_str = "32F";
            break;
        case 6:
            type_str = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number / 8) + 1;

    std::stringstream type;
    type << "CV_" << type_str << "C" << channel;

    return type.str();
}

} // end anonymous namespace

/// Return a vector of descriptor shared pointers
std::vector<vital::descriptor_sptr>
descriptor_set
::descriptors() const
{
  std::vector<vital::descriptor_sptr> desc;
  const unsigned num_desc = data_.rows;
  /// \cond DoxygenSuppress
#define CONVERT_CASE(T) \
  case cv::DataType<T>::type: \
  for( unsigned i=0; i<num_desc; ++i ) \
  { \
    desc.push_back(ocv_to_vital_descriptor<T>(data_.row(i))); \
  } \
  break

  switch(data_.type())
  {
  APPLY_TO_TYPES(CONVERT_CASE);
  default:
    VITAL_THROW( vital::invalid_value, "No case to handle OpenCV descriptors of type "
                               + cv_type_to_string(data_.type()));
  }
#undef CONVERT_CASE
  /// \endcond
  return desc;
}

/// Convert any descriptor set to an OpenCV cv::Mat
cv::Mat
descriptors_to_ocv_matrix(const vital::descriptor_set& desc_set)
{
  // if the descriptor set already contains a cv::Mat representation
  // then return the existing matrix
  if( const ocv::descriptor_set* d =
          dynamic_cast<const ocv::descriptor_set*>(&desc_set) )
  {
    return d->ocv_desc_matrix();
  }
  std::vector<vital::descriptor_sptr> desc = desc_set.descriptors();
  if( desc.empty() || !desc[0] )
  {
    return cv::Mat();
  }
  /// \cond DoxygenSuppress
#define CONVERT_CASE(T) \
  if( dynamic_cast<const vital::descriptor_array_of<T>*>(desc[0].get()) ) \
  { \
    return vital_descriptors_to_ocv<T>(desc); \
  }
  APPLY_TO_TYPES(CONVERT_CASE);
#undef CONVERT_CASE
  /// \endcond
  return cv::Mat();
}

/// Return the descriptor at the specified index
vital::descriptor_sptr
descriptor_set
::at( size_t index )
{

  /// \cond DoxygenSuppress
#define CONVERT_CASE(T) \
  case cv::DataType<T>::type:                   \
  return ocv_to_vital_descriptor<T>( data_.row(index) );

  switch(data_.type())
  {
    APPLY_TO_TYPES(CONVERT_CASE);
  default:
    VITAL_THROW( vital::invalid_value, "No case to handle OpenCV descriptors of type "
                               + cv_type_to_string(data_.type()));
  }
#undef CONVERT_CASE
}

/// Return the descriptor at the specified index (const)
vital::descriptor_sptr const
descriptor_set
::at( size_t index ) const
{

  /// \cond DoxygenSuppress
#define CONVERT_CASE(T)                         \
  case cv::DataType<T>::type:                   \
    return ocv_to_vital_descriptor<T>( data_.row(index) );

  switch(data_.type())
  {
    APPLY_TO_TYPES(CONVERT_CASE);
  default:
    VITAL_THROW( vital::invalid_value, "No case to handle OpenCV descriptors of type "
                               + cv_type_to_string(data_.type()));
  }
#undef CONVERT_CASE
}

/// Next-descriptor generation function.
descriptor_set::iterator::next_value_func_t
descriptor_set
::get_iter_next_func()
{
  size_t row_counter = 0;
  // Variable for copy into the lambda instance to hold the current row
  // descriptor reference.
  vital::descriptor_sptr d_sptr;
  return [row_counter,d_sptr,this] () mutable ->iterator::reference {
    if( row_counter >= size() )
    {
      VITAL_THROW( vital::stop_iteration_exception, "descriptor_set" );
    }

  /// \cond DoxygenSuppress
#define CONVERT_CASE(T) \
    case cv::DataType<T>::type:                                         \
      d_sptr = ocv_to_vital_descriptor<T>( data_.row(row_counter++) ); \
      break;

  switch(data_.type())
  {
    APPLY_TO_TYPES(CONVERT_CASE);
  default:
    VITAL_THROW( vital::invalid_value, "No case to handle OpenCV descriptors of type "
                               + cv_type_to_string(data_.type()));
  }
#undef CONVERT_CASE
    return d_sptr;
  };
}

/// Next-descriptor generation function. (const)
descriptor_set::const_iterator::next_value_func_t
descriptor_set
::get_const_iter_next_func() const
{
  size_t row_counter = 0;
  // Variable for copy into the lambda instance to hold the current row
  // descriptor reference.
  vital::descriptor_sptr d_sptr;
  return [row_counter,d_sptr,this] () mutable ->const_iterator::reference {
    if( row_counter >= size() )
    {
      VITAL_THROW( vital::stop_iteration_exception, "descriptor_set" );
    }

  /// \cond DoxygenSuppress
#define CONVERT_CASE(T)                         \
  case cv::DataType<T>::type:                   \
    d_sptr = ocv_to_vital_descriptor<T>( data_.row(row_counter++) ); \
    break;

  switch(data_.type())
  {
    APPLY_TO_TYPES(CONVERT_CASE);
  default:
    VITAL_THROW( vital::invalid_value, "No case to handle OpenCV descriptors of type "
                               + cv_type_to_string(data_.type()));
  }
#undef CONVERT_CASE
    return d_sptr;
  };
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
