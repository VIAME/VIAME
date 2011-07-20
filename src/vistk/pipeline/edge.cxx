/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "edge.h"
#include "edge_exception.h"

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

#include <queue>

/**
 * \file edge.cxx
 *
 * \brief Implementation of the \ref edge class.
 */

namespace vistk
{

class edge::priv
{
  public:
    priv();
    ~priv();

    bool required;

    process_t upstream;
    process_t downstream;

    std::queue<edge_datum_t> q;

    boost::condition_variable cond_have_data;

    mutable boost::mutex mutex;
};

edge
::~edge()
{
}

bool
edge
::makes_dependency() const
{
  return true;
}

bool
edge
::has_data() const
{
  return (datum_count() != 0);
}

bool
edge
::full_of_data() const
{
  return false;
}

size_t
edge
::datum_count() const
{
  boost::unique_lock<boost::mutex> lock(d->mutex);

  (void)lock;

  return d->q.size();
}

void
edge
::push_datum(edge_datum_t const& datum)
{
  boost::unique_lock<boost::mutex> lock(d->mutex);

  (void)lock;

  d->q.push(datum);
}

edge_datum_t
edge
::get_datum()
{
  edge_datum_t const dat = peek_datum();

  pop_datum();

  return dat;
}

edge_datum_t
edge
::peek_datum()
{
  boost::unique_lock<boost::mutex> lock(d->mutex);

  while (!has_data())
  {
    d->cond_have_data.wait(lock);
  }

  return d->q.front();
}

void
edge
::pop_datum()
{
  boost::unique_lock<boost::mutex> lock(d->mutex);

  while (!has_data())
  {
    d->cond_have_data.wait(lock);
  }

  d->q.pop();
}

void
edge
::set_required_by_downstream(bool required)
{
  d->required = required;
}

bool
edge
::required_by_downstream() const
{
  return d->required;
}

void
edge
::set_upstream_process(process_t process)
{
  if (!process)
  {
    throw null_process_connection();
  }

  if (d->upstream)
  {
    throw input_already_connected(d->upstream->name(), process->name());
  }

  d->upstream = process;
}

void
edge
::set_downstream_process(process_t process)
{
  if (!process)
  {
    throw null_process_connection();
  }

  if (d->downstream)
  {
    throw output_already_connected(d->downstream->name(), process->name());
  }

  d->downstream = process;
}

edge
::edge(config_t const& /*config*/)
{
}

edge::priv
::priv()
  : required(true)
{
}

edge::priv
::~priv()
{
}

} // end namespace vistk
