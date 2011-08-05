/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "schedule.h"
#include "schedule_exception.h"

/**
 * \file schedule.cxx
 *
 * \brief Implementation of the base class for \link vistk::schedule schedules\endlink.
 */

namespace vistk
{

class schedule::priv
{
  public:
    priv(pipeline_t const& pipeline);
    ~priv();

    pipeline_t const p;
};

schedule
::~schedule()
{
}

schedule
::schedule(config_t const& config, pipeline_t const& pipe)
{
  if (!config)
  {
    throw null_schedule_config_exception();
  }

  if (!pipe)
  {
    throw null_schedule_pipeline_exception();
  }

  d = boost::shared_ptr<priv>(new priv(pipe));
}

pipeline_t
schedule
::pipeline() const
{
  return d->p;
}

schedule::priv
::priv(pipeline_t const& pipe)
  : p(pipe)
{
}

schedule::priv
::~priv()
{
}

}
