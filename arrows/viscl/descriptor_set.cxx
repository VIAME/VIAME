// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "descriptor_set.h"

#include <viscl/core/manager.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// Return a vector of descriptor shared pointers
std::vector<vital::descriptor_sptr>
descriptor_set
::descriptors() const
{
  using namespace arrows;
  std::vector<vital::descriptor_sptr> desc;

  cl_int4 *buf = new cl_int4[data_.len()];

  viscl::cl_queue_t queue = viscl::manager::inst()->create_queue();
  queue->enqueueReadBuffer(*data_().get(), CL_TRUE, 0, data_.mem_size(), buf);
  queue->finish();

  for (unsigned int i = 0; i < data_.len(); i++)
  {
    vital::descriptor_fixed<int,4> *d = new vital::descriptor_fixed<int,4>;
    memcpy(d->raw_data(), &buf[i].s, sizeof(int)*4);
    desc.push_back(vital::descriptor_sptr(d));
  }

  delete [] buf;

  return desc;
}

/// Convert a descriptor set to a VisCL descriptor set must be <int,4>
viscl::buffer
descriptors_to_viscl(const vital::descriptor_set& desc_set)
{
  if( const vcl::descriptor_set* m_viscl =
          dynamic_cast<const vcl::descriptor_set*>(&desc_set) )
  {
    return m_viscl->viscl_descriptors();
  }

  //viscl cannot take an arbitrary descriptor so this function
  //only checks for <int,4> descriptors
  std::vector<cl_int4> viscl_descr;
  std::vector<vital::descriptor_sptr> descriptors = desc_set.descriptors();
  for (unsigned int i = 0; i < descriptors.size(); i++)
  {
    //check if type is <int,4> if not we are done
    if ( const vital::descriptor_fixed<int,4> * dfixed =
          dynamic_cast<const vital::descriptor_fixed<int,4> *>(descriptors[i].get()) )
    {
      cl_int4 d;
      memcpy(&d.s, dfixed->raw_data(), sizeof(int)*4);
      viscl_descr.push_back(d);
    }
    else
    {
      break;
    }
  }

  if (viscl_descr.size() == descriptors.size())
  {
    viscl::buffer buf = viscl::manager::inst()->create_buffer<cl_int4>(CL_MEM_READ_WRITE, viscl_descr.size());
    viscl::cl_queue_t queue = viscl::manager::inst()->create_queue();
    queue->enqueueWriteBuffer(*buf().get(), CL_TRUE, 0, buf.mem_size(), &viscl_descr[0]);
    queue->finish();
    return buf;
  }

  //TODO: throw exception
  return viscl::buffer();
}

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver
