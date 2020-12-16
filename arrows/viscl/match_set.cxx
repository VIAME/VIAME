// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <algorithm>

#include <arrows/viscl/match_set.h>

#include <viscl/core/manager.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// Return the number of matches in the set
size_t
match_set
::size() const
{
  std::vector<int> viscl_matches(data_.len());
  viscl::cl_queue_t queue = viscl::manager::inst()->create_queue();
  queue->enqueueReadBuffer(*data_().get(), CL_TRUE, 0, data_.mem_size(), &viscl_matches[0]);
  queue->finish();

  size_t count = 0;
  for (unsigned int i = 0; i < viscl_matches.size(); i++)
  {
    if (viscl_matches[i] > -1)
    {
      count++;
    }
  }

  return count;
}

/// Return a vector of matching indices
std::vector<vital::match>
match_set
::matches() const
{
  std::vector<vital::match> m;

  std::vector<int> viscl_matches(data_.len());
  viscl::cl_queue_t queue = viscl::manager::inst()->create_queue();
  queue->enqueueReadBuffer(*data_().get(), CL_TRUE, 0, data_.mem_size(), &viscl_matches[0]);
  queue->finish();

  for (unsigned int i = 0; i < viscl_matches.size(); i++)
  {
    if (viscl_matches[i] > -1)
    {
      m.push_back(vital::match(viscl_matches[i], i));
    }
  }

  return m;
}

/// Convert any match set to VisCL matches
viscl::buffer
matches_to_viscl(const vital::match_set& m_set)
{
  if( const vcl::match_set* m_viscl =
          dynamic_cast<const vcl::match_set*>(&m_set) )
  {
    return m_viscl->viscl_matches();
  }

  const std::vector<vital::match> mats = m_set.matches();

  unsigned int maxindex = 0;
  for (unsigned int i = 0; i < mats.size(); i++)
  {
    if (mats[i].second > maxindex)
    {
      maxindex = mats[i].second;
    }
  }

  std::vector<int> buf(maxindex + 1, -1);
  for (unsigned int i = 0; i < mats.size(); i++)
  {
    buf[mats[i].second] = mats[i].first;
  }

  viscl::buffer viscl_data = viscl::manager::inst()->create_buffer<int>(CL_MEM_READ_WRITE, buf.size());
  viscl::cl_queue_t queue = viscl::manager::inst()->create_queue();
  queue->enqueueWriteBuffer(*viscl_data().get(), CL_TRUE, 0, viscl_data.mem_size(), &buf[0]);
  queue->finish();

  return viscl_data;
}

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver
