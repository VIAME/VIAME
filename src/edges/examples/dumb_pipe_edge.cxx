/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "dumb_pipe_edge.h"

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

#include <queue>

namespace vistk
{

class dumb_pipe_edge::priv
{
  public:
    priv();
    ~priv();

    std::queue<edge_datum_t> q;

    boost::condition_variable cond_have_data;

    mutable boost::mutex mutex;
};

dumb_pipe_edge
::dumb_pipe_edge(config_t const& config)
  : edge(config)
{
}

dumb_pipe_edge
::~dumb_pipe_edge()
{
}

bool
dumb_pipe_edge
::full_of_data() const
{
  return false;
}

size_t
dumb_pipe_edge
::datum_count() const
{
  boost::unique_lock<boost::mutex> lock(d->mutex);

  (void)lock;

  return d->q.size();
}

void
dumb_pipe_edge
::push_datum(edge_datum_t const& datum)
{
  boost::unique_lock<boost::mutex> lock(d->mutex);

  (void)lock;

  d->q.push(datum);
}

edge_datum_t
dumb_pipe_edge
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
dumb_pipe_edge
::pop_datum()
{
  boost::unique_lock<boost::mutex> lock(d->mutex);

  while (!has_data())
  {
    d->cond_have_data.wait(lock);
  }

  d->q.pop();
}

dumb_pipe_edge::priv
::priv()
{
}

dumb_pipe_edge::priv
::~priv()
{
}

} // end namespace vistk
