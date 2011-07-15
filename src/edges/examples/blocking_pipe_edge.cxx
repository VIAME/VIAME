/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "blocking_pipe_edge.h"

#include <vistk/pipeline/config.h>

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

#include <queue>

namespace vistk
{

class blocking_pipe_edge::priv
{
  public:
    priv(size_t size);
    ~priv();

    void flip_stack();

    size_t const max_size;

    std::queue<edge_datum_t> in_q;
    std::queue<edge_datum_t> out_q;

    boost::condition_variable cond_have_data;
    boost::condition_variable cond_have_space;

    mutable boost::mutex in_mutex;
    mutable boost::mutex out_mutex;
};

blocking_pipe_edge
::blocking_pipe_edge(config_t const& config)
  : edge(config)
{
  size_t const max_size = config->get_value<size_t>("max_size", 15);

  d = boost::shared_ptr<priv>(new priv(max_size));
}

blocking_pipe_edge
::~blocking_pipe_edge()
{
}

bool
blocking_pipe_edge
::full_of_data() const
{
  return (d->max_size == datum_count());
}

size_t
blocking_pipe_edge
::datum_count() const
{
  boost::unique_lock<boost::mutex> in_lock(d->in_mutex);
  boost::unique_lock<boost::mutex> out_lock(d->out_mutex);

  (void)in_lock;
  (void)out_lock;

  return (d->in_q.size() + d->out_q.size());
}

void
blocking_pipe_edge
::push_datum(edge_datum_t const& datum)
{
  boost::unique_lock<boost::mutex> in_lock(d->in_mutex);

  while (full_of_data())
  {
    d->cond_have_space.wait(in_lock);
  }

  d->in_q.push(datum);

  d->cond_have_data.notify_one();
}

edge_datum_t
blocking_pipe_edge
::peek_datum()
{
  boost::unique_lock<boost::mutex> out_lock(d->out_mutex);

  while (!has_data())
  {
    d->cond_have_data.wait(out_lock);
  }

  if (!d->out_q.size())
  {
    d->flip_stack();
  }

  return d->out_q.front();
}

void
blocking_pipe_edge
::pop_datum()
{
  boost::unique_lock<boost::mutex> out_lock(d->out_mutex);

  while (!has_data())
  {
    d->cond_have_data.wait(out_lock);
  }

  if (!d->out_q.size())
  {
    d->flip_stack();
  }

  d->out_q.pop();

  d->cond_have_space.notify_one();
}

blocking_pipe_edge::priv
::priv(size_t size)
  : max_size(size)
{
}

blocking_pipe_edge::priv
::~priv()
{
}

void
blocking_pipe_edge::priv
::flip_stack()
{
  boost::unique_lock<boost::mutex> in_lock(in_mutex);

  (void)in_lock;

  while (in_q.size())
  {
    out_q.push(in_q.front());
    in_q.pop();
  }
}

} // end namespace vistk
