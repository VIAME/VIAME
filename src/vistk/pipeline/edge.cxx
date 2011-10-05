/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "edge.h"
#include "edge_exception.h"

#include "stamp.h"
#include "types.h"

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

#include <queue>

/**
 * \file edge.cxx
 *
 * \brief Implementation of the \link vistk::edge edge\endlink class.
 */

namespace vistk
{

class edge::priv
{
  public:
    priv();
    ~priv();

    bool has_data() const;
    void complete_check() const;

    bool required;
    bool downstream_complete;

    process_t upstream;
    process_t downstream;

    std::queue<edge_datum_t> q;

    boost::condition_variable cond_have_data;

    mutable boost::mutex mutex;
    mutable boost::mutex complete_mutex;
};

edge
::edge(config_t const& config)
  : d(new priv)
{
  if (!config)
  {
    throw null_edge_config_exception();
  }
}

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
  boost::mutex::scoped_lock const lock(d->mutex);

  (void)lock;

  return d->has_data();
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
  boost::mutex::scoped_lock const lock(d->mutex);

  (void)lock;

  return d->q.size();
}

void
edge
::push_datum(edge_datum_t const& datum)
{
  {
    boost::mutex::scoped_lock const lock(d->complete_mutex);

    (void)lock;

    if (d->downstream_complete)
    {
      return;
    }
  }

  {
    boost::mutex::scoped_lock const lock(d->mutex);

    (void)lock;

    d->q.push(datum);
  }

  d->cond_have_data.notify_one();
}

edge_datum_t
edge
::get_datum()
{
  d->complete_check();

  boost::mutex::scoped_lock lock(d->mutex);

  while (!d->has_data())
  {
    d->cond_have_data.wait(lock);
  }

  edge_datum_t const dat = d->q.front();

  d->q.pop();

  return dat;
}

edge_datum_t
edge
::peek_datum()
{
  d->complete_check();

  boost::mutex::scoped_lock lock(d->mutex);

  while (!d->has_data())
  {
    d->cond_have_data.wait(lock);
  }

  return d->q.front();
}

void
edge
::pop_datum()
{
  d->complete_check();

  boost::mutex::scoped_lock lock(d->mutex);

  while (!d->has_data())
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
::mark_downstream_as_complete()
{
  boost::mutex::scoped_lock complete_lock(d->complete_mutex);

  (void)complete_lock;

  d->downstream_complete = true;

  boost::mutex::scoped_lock const lock(d->mutex);

  (void)lock;

  while (d->q.size())
  {
    d->q.pop();
  }
}

bool
edge
::is_downstream_complete() const
{
  boost::mutex::scoped_lock const lock(d->complete_mutex);

  (void)lock;

  return d->downstream_complete;
}

void
edge
::set_upstream_process(process_t process)
{
  if (!process)
  {
    throw null_process_connection_exception();
  }

  if (d->upstream)
  {
    throw input_already_connected_exception(d->upstream->name(), process->name());
  }

  d->upstream = process;
}

void
edge
::set_downstream_process(process_t process)
{
  if (!process)
  {
    throw null_process_connection_exception();
  }

  if (d->downstream)
  {
    throw output_already_connected_exception(d->downstream->name(), process->name());
  }

  d->downstream = process;
}

bool
operator == (edge_datum_t const& a, edge_datum_t const& b)
{
  return (( a.get<0>() ==  b.get<0>()) &&
          (*a.get<1>() == *b.get<1>()));
}

edge::priv
::priv()
  : required(true)
  , downstream_complete(false)
{
}

edge::priv
::~priv()
{
}

bool
edge::priv
::has_data() const
{
  return (q.size() != 0);
}

void
edge::priv
::complete_check() const
{
  boost::mutex::scoped_lock const lock(complete_mutex);

  (void)lock;

  if (downstream_complete)
  {
    throw datum_requested_after_complete();
  }
}

}
