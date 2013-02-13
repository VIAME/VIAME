/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "edge.h"
#include "edge_exception.h"

#include "stamp.h"
#include "types.h"

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/weak_ptr.hpp>

#include <deque>

/**
 * \file edge.cxx
 *
 * \brief Implementation of the \link vistk::edge edge\endlink class.
 */

namespace vistk
{

edge_datum_t
::edge_datum_t()
  : datum()
  , stamp()
{
}

edge_datum_t
::edge_datum_t(datum_t const& datum_, stamp_t const& stamp_)
  : datum(datum_)
  , stamp(stamp_)
{
}

edge_datum_t
::~edge_datum_t()
{
}

bool
edge_datum_t
::operator == (edge_datum_t const& rhs) const
{
  return (( datum ==  rhs.datum) &&
          (*stamp == *rhs.stamp));
}

config::key_t const edge::config_dependency = config::key_t("_dependency");
config::key_t const edge::config_capacity = config::key_t("capacity");

class edge::priv
{
  public:
    priv(bool depends_, size_t capacity_);
    ~priv();

    typedef boost::weak_ptr<process> process_ref_t;

    bool has_data() const;
    bool full_of_data() const;
    void complete_check() const;

    bool const depends;
    size_t const capacity;
    bool downstream_complete;

    process_ref_t upstream;
    process_ref_t downstream;

    typedef std::deque<edge_datum_t> edge_queue_t;

    edge_queue_t q;

    boost::condition_variable_any cond_have_data;
    boost::condition_variable_any cond_have_space;

    typedef boost::shared_mutex mutex_t;
    typedef boost::shared_lock<mutex_t> shared_lock_t;
    typedef boost::upgrade_lock<mutex_t> upgrade_lock_t;
    typedef boost::unique_lock<mutex_t> unique_lock_t;
    typedef boost::upgrade_to_unique_lock<mutex_t> upgrade_to_unique_lock_t;

    mutable mutex_t mutex;
    mutable mutex_t complete_mutex;
};

edge
::edge(config_t const& config)
  : d()
{
  if (!config)
  {
    throw null_edge_config_exception();
  }

  bool const depends = config->get_value<bool>(config_dependency, true);
  size_t const capacity = config->get_value<size_t>(config_capacity, 0);

  d.reset(new priv(depends, capacity));
}

edge
::~edge()
{
}

bool
edge
::makes_dependency() const
{
  return d->depends;
}

bool
edge
::has_data() const
{
  priv::shared_lock_t const lock(d->mutex);

  (void)lock;

  return d->has_data();
}

bool
edge
::full_of_data() const
{
  priv::shared_lock_t const lock(d->mutex);

  (void)lock;

  return d->full_of_data();
}

size_t
edge
::datum_count() const
{
  priv::shared_lock_t const lock(d->mutex);

  (void)lock;

  return d->q.size();
}

void
edge
::push_datum(edge_datum_t const& datum)
{
  {
    priv::shared_lock_t const lock(d->complete_mutex);

    (void)lock;

    if (d->downstream_complete)
    {
      return;
    }
  }

  {
    priv::upgrade_lock_t lock(d->mutex);

    while (d->full_of_data())
    {
      d->cond_have_space.wait(lock);
    }

    {
      priv::upgrade_to_unique_lock_t const write_lock(lock);

      (void)write_lock;

      d->q.push_back(datum);
    }
  }

  d->cond_have_data.notify_one();
}

edge_datum_t
edge
::get_datum()
{
  d->complete_check();

  edge_datum_t dat;

  {
    priv::upgrade_lock_t lock(d->mutex);

    while (!d->has_data())
    {
      d->cond_have_data.wait(lock);
    }

    dat = d->q.front();

    {
      priv::upgrade_to_unique_lock_t const write_lock(lock);

      (void)write_lock;

      d->q.pop_front();
    }
  }

  d->cond_have_space.notify_one();

  return dat;
}

edge_datum_t
edge
::peek_datum(size_t idx) const
{
  d->complete_check();

  priv::shared_lock_t lock(d->mutex);

  while (d->q.size() <= idx)
  {
    d->cond_have_data.wait(lock);
  }

  return d->q.at(idx);
}

void
edge
::pop_datum()
{
  d->complete_check();

  {
    priv::upgrade_lock_t lock(d->mutex);

    while (!d->has_data())
    {
      d->cond_have_data.wait(lock);
    }

    {
      priv::upgrade_to_unique_lock_t const write_lock(lock);

      (void)write_lock;

      d->q.pop_front();
    }
  }

  d->cond_have_space.notify_one();
}

void
edge
::mark_downstream_as_complete()
{
  priv::unique_lock_t const complete_lock(d->complete_mutex);
  priv::unique_lock_t const lock(d->mutex);

  (void)complete_lock;
  (void)lock;

  d->downstream_complete = true;

  while (!d->q.empty())
  {
    d->q.pop_front();
  }

  d->cond_have_space.notify_one();
}

bool
edge
::is_downstream_complete() const
{
  priv::shared_lock_t const lock(d->complete_mutex);

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

  if (!d->upstream.expired())
  {
    process_t const up = d->upstream.lock();

    throw input_already_connected_exception(up->name(), process->name());
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

  if (!d->downstream.expired())
  {
    process_t const down = d->downstream.lock();

    throw output_already_connected_exception(down->name(), process->name());
  }

  d->downstream = process;
}

edge::priv
::priv(bool depends_, size_t capacity_)
  : depends(depends_)
  , capacity(capacity_)
  , downstream_complete(false)
  , upstream()
  , downstream()
  , q()
  , cond_have_data()
  , cond_have_space()
  , mutex()
  , complete_mutex()
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
  return !q.empty();
}

bool
edge::priv
::full_of_data() const
{
  if (!capacity)
  {
    return false;
  }

  return (q.size() == capacity);
}

void
edge::priv
::complete_check() const
{
  shared_lock_t const lock(complete_mutex);

  (void)lock;

  if (downstream_complete)
  {
    throw datum_requested_after_complete();
  }
}

}
