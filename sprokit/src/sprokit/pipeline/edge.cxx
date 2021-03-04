// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "edge.h"
#include "edge_exception.h"

#include "stamp.h"
#include "types.h"

#include <vital/logger/logger.h>

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include <deque>

/**
 * \file edge.cxx
 *
 * \brief Implementation of the \link sprokit::edge edge\endlink class.
 */

// Check to see if there is an external specification for default edge
// capacity
#if !defined SPROKIT_DEFAULT_EDGE_CAPACITY
#  define SPROKIT_DEFAULT_EDGE_CAPACITY 10 // default size
#endif

namespace sprokit {

// ------------------------------------------------------------------
edge_datum_t
::edge_datum_t()
  : datum()
  , stamp()
{
}

// ------------------------------------------------------------------
edge_datum_t
::edge_datum_t(datum_t const& datum_, stamp_t const& stamp_)
  : datum(datum_)
  , stamp(stamp_)
{
}

// ------------------------------------------------------------------
edge_datum_t
::~edge_datum_t()
{
}

template <typename T>
static bool pointers_equal(T const& a, T const& b);

// ------------------------------------------------------------------
bool
edge_datum_t
::operator == (edge_datum_t const& rhs) const
{
  return (pointers_equal(datum, rhs.datum) &&
          pointers_equal(stamp, rhs.stamp));
}

// This config parameter is used internally to signal that the edge
// has no dependency. See process::flag_input_nodep for additional
// description.
kwiver::vital::config_block_key_t const edge::config_dependency = kwiver::vital::config_block_key_t("_dependency");
kwiver::vital::config_block_key_t const edge::config_capacity   = kwiver::vital::config_block_key_t("capacity");
kwiver::vital::config_block_key_t const edge::config_blocking   = kwiver::vital::config_block_key_t("blocking");

// ==================================================================
class edge::priv
{
  public:
    priv(bool depends_, size_t capacity_, bool blocking_);
    ~priv();

    typedef std::weak_ptr<process> process_ref_t;

    bool full_of_data() const;
    void complete_check() const;

    bool push(edge_datum_t const& datum, kwiver::vital::optional<duration_t> const& duration = kwiver::vital::nullopt);
    kwiver::vital::optional<edge_datum_t> pop(kwiver::vital::optional<duration_t> const& duration = kwiver::vital::nullopt);

    /// This flag indicates that this edge connection should or should
    /// not imply a dependency. Generally set to false if a backwards
    /// edge.
    bool const depends;

    /// Size of the buffer in this edge.
    size_t const capacity;

    /// Set to indicate if this edge will block if its buffer is full.
    bool const blocking;

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

    kwiver::vital::logger_handle_t m_logger;
};

// ==================================================================
edge
::edge(kwiver::vital::config_block_sptr const& config)
  : d()
{
  if (!config)
  {
    VITAL_THROW( null_edge_config_exception );
  }

  bool const depends    = config->get_value<bool>(config_dependency, true);
  size_t const capacity = config->get_value<size_t>(config_capacity, SPROKIT_DEFAULT_EDGE_CAPACITY );
  bool const blocking   = config->get_value<bool>(config_blocking, true);

  d.reset(new priv(depends, capacity, blocking));

  if ( 0 != capacity || ! blocking )
  {
    LOG_DEBUG( d->m_logger, "Edge capacity set to: " << capacity
               << "   " << (blocking ? "" : "non-" ) << "blocking: ");
  }
}

// ------------------------------------------------------------------
edge
::~edge()
{
}

// ------------------------------------------------------------------
bool
edge
::makes_dependency() const
{
  return d->depends;
}

// ------------------------------------------------------------------
bool
edge
::has_data() const
{
  priv::shared_lock_t const lock(d->mutex);

  (void)lock;

  return !d->q.empty();
}

// ------------------------------------------------------------------
bool
edge
::full_of_data() const
{
  priv::shared_lock_t const lock(d->mutex);

  (void)lock;

  return d->full_of_data();
}

// ------------------------------------------------------------------
size_t
edge
::datum_count() const
{
  priv::shared_lock_t const lock(d->mutex);

  (void)lock;

  return d->q.size();
}

// ------------------------------------------------------------------
void
edge
::push_datum(edge_datum_t const& datum)
{
  // If non blocking, set duration to zero for no delay
  // If the datum is a control element, then we always want to block.
  // We can lose data but not control messages.
  if ( ! d->blocking && ( datum.datum->type() == datum::data ) )
  {
    d->push( datum, duration_t( 0 ) );
  }
  else
  {
    d->push( datum );
  }
}

// ------------------------------------------------------------------
edge_datum_t
edge
::get_datum()
{
  return *d->pop();
}

// ------------------------------------------------------------------
edge_datum_t
edge
::peek_datum(size_t idx) const
{
  d->complete_check();

  priv::shared_lock_t lock(d->mutex);

  d->cond_have_data.wait(lock,
      boost::bind(&priv::edge_queue_t::size, &d->q) > idx);

  return d->q.at(idx);
}

// ------------------------------------------------------------------
void
edge
::pop_datum()
{
  d->complete_check();

  {
    priv::upgrade_lock_t lock(d->mutex);

    d->cond_have_data.wait(lock,
        !boost::bind(&priv::edge_queue_t::empty, &d->q));

    {
      priv::upgrade_to_unique_lock_t const write_lock(lock);

      (void)write_lock;

      d->q.pop_front();
    }
  }

  d->cond_have_space.notify_one();
}

// ------------------------------------------------------------------
bool
edge
::try_push_datum(edge_datum_t const& datum, duration_t const& duration)
{
  return d->push(datum, duration);
}

// ------------------------------------------------------------------
kwiver::vital::optional<edge_datum_t>
edge
::try_get_datum(duration_t const& duration)
{
  return d->pop(duration);
}

// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
bool
edge
::is_downstream_complete() const
{
  priv::shared_lock_t const lock(d->complete_mutex);

  (void)lock;

  return d->downstream_complete;
}

// ------------------------------------------------------------------
void
edge
::set_upstream_process(process_t process)
{
  if (!process)
  {
    VITAL_THROW( null_process_connection_exception );
  }

  if (!d->upstream.expired())
  {
    process_t const up = d->upstream.lock();

    VITAL_THROW( input_already_connected_exception,
                 up->name(), process->name());
  }

  d->upstream = process;
}

// ------------------------------------------------------------------
void
edge
::set_downstream_process(process_t process)
{
  if (!process)
  {
    VITAL_THROW( null_process_connection_exception );
  }

  if (!d->downstream.expired())
  {
    process_t const down = d->downstream.lock();

    VITAL_THROW( output_already_connected_exception,
                 down->name(), process->name());
  }

  d->downstream = process;
}

// ==================================================================
edge::priv
::priv(bool depends_, size_t capacity_, bool blocking_)
  : depends(depends_)
  , capacity(capacity_)
  , blocking(blocking_)
  , downstream_complete(false)
  , upstream()
  , downstream()
  , q()
  , cond_have_data()
  , cond_have_space()
  , mutex()
  , complete_mutex()
  , m_logger( kwiver::vital::get_logger( "sprokit.edge" ))
{
}

// ------------------------------------------------------------------
edge::priv
::~priv()
{
}

// ------------------------------------------------------------------
bool
edge::priv
::full_of_data() const
{
  if (!capacity)
  {
    return false;
  }

  return (capacity <= q.size());
}

// ------------------------------------------------------------------
void
edge::priv
::complete_check() const
{
  shared_lock_t const lock(complete_mutex);

  (void)lock;

  if (downstream_complete)
  {
    VITAL_THROW( datum_requested_after_complete );
  }
}

// ------------------------------------------------------------------
bool
edge::priv
::push(edge_datum_t const& datum, kwiver::vital::optional<duration_t> const& duration)
{
  {
    shared_lock_t const lock(complete_mutex);

    (void)lock;

    // If downstream process has marked itself as complete, do nothing
    if (downstream_complete)
    {
      return true;
    }
  }

  {
    upgrade_lock_t lock(mutex);
    boost::function<bool ()> const predicate = !boost::bind(&sprokit::edge::priv::full_of_data, this);

    if (duration)
    {
      // Wait for specified duration before giving up
      if (!cond_have_space.wait_for(lock, *duration, predicate))
      {
        return false;
      }
    }
    else
    {
      cond_have_space.wait(lock, predicate);
    }

    {
      upgrade_to_unique_lock_t const write_lock(lock);

      (void)write_lock;

      q.push_back(datum);
    }
  }

  cond_have_data.notify_one();

  return true;
}

// ------------------------------------------------------------------
kwiver::vital::optional<edge_datum_t>
edge::priv
::pop(kwiver::vital::optional<duration_t> const& duration)
{
  complete_check();

  edge_datum_t dat;

  {
    upgrade_lock_t lock(mutex);
    boost::function<bool ()> const predicate = !boost::bind(&edge_queue_t::empty, &q);

    if (duration)
    {
      if (!cond_have_data.wait_for(lock, *duration, predicate))
      {
        return kwiver::vital::nullopt;
      }
    }
    else
    {
      cond_have_data.wait(lock, predicate);
    }

    dat = q.front();

    {
      upgrade_to_unique_lock_t const write_lock(lock);

      (void)write_lock;

      q.pop_front();
    }
  }

  cond_have_space.notify_one();

  return dat;
}

// ------------------------------------------------------------------
template <typename T>
bool
pointers_equal(T const& a, T const& b)
{
  if (a == b)
  {
    return true;
  }

  if (!a || !b)
  {
    return false;
  }

  return (*a == *b);
}

}
