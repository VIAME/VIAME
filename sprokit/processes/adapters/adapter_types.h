// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file adapter_types.h
 * \brief Interface file for adapter types.
 *
 * This file contains a set of types that can be used to declare
 * useful objects. They are in a separate file to reduce coupling in
 * the interface (header) files that use these types.
 */

#ifndef KWIVER_ADAPTER_ADAPTER_TYPES_H
#define KWIVER_ADAPTER_ADAPTER_TYPES_H

#include <sprokit/pipeline/process.h>

#include <memory>
#include <map>

namespace kwiver {
namespace vital {

template <class T> class bounded_buffer;

}

namespace adapter{

class adapter_data_set;
typedef std::shared_ptr< adapter_data_set > adapter_data_set_t;
typedef std::shared_ptr< kwiver::vital::bounded_buffer< kwiver::adapter::adapter_data_set_t > > interface_ref_t;
typedef std::map< sprokit::process::port_t, sprokit::process::port_info_t > ports_info_t;

} } // end namespace

#endif // KWIVER_ADAPTER_ADAPTER_TYPES_H
