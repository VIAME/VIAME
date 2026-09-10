// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_TEST_INTERFACE_INTERFACE_H
#define VITAL_TEST_INTERFACE_INTERFACE_H

#include <viame/algorithm_framework/plugin/pluggable.h>

namespace kwiver::vital {

class say : public pluggable
{
public:
  ~say() override = default;

  static std::string interface_name() { return "say"; }

  /**
   * Say something
   * @return String of what way say.
   */
  virtual std::string says() = 0;
};

typedef std::shared_ptr< say > say_sptr;

} // namespace kwiver::vital

#endif // VITAL_TEST_INTERFACE_INTERFACE_H
