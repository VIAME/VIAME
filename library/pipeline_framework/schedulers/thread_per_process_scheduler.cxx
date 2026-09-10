// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "thread_per_process_scheduler.h"

#include <viame/algorithm_framework/config/config_block.h>

#include <viame/pipeline_framework/datum.h>
#include <viame/pipeline_framework/edge.h>
#include <viame/pipeline_framework/pipeline.h>
#include <viame/pipeline_framework/scheduler_exception.h>
#include <viame/pipeline_framework/utils.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <vector>

/**
 * \file thread_per_process_scheduler.cxx
 *
 * \brief Implementation of the thread-per-process scheduler.
 */

namespace sprokit
{

class thread_per_process_scheduler::priv
{
  public:
    priv();
    ~priv();

    void run_process(process_t const& process,
                     kwiver::vital::logger_handle_t logger);

    std::vector<std::thread> process_threads;
    std::atomic<bool> m_stop_requested{false};

    typedef std::shared_mutex mutex_t;
    typedef std::shared_lock<mutex_t> shared_lock_t;

    mutable mutex_t m_pause_mutex;

    void join_all()
    {
      for (auto& t : process_threads)
      {
        if (t.joinable())
        {
          t.join();
        }
      }
      process_threads.clear();
    }

    void request_stop()
    {
      m_stop_requested = true;
    }

    bool stop_requested() const
    {
      return m_stop_requested;
    }

    // Error tracking
    std::atomic<bool> m_error_occurred{false};
    std::mutex m_error_mutex;
    std::string m_error_message;
    std::exception_ptr m_exception_ptr;
};

// ------------------------------------------------------------------
thread_per_process_scheduler
::thread_per_process_scheduler(pipeline_t const& pipe,
                               kwiver::vital::config_block_sptr const& config)
  : scheduler(pipe, config)
  , d(new priv)
{
  m_logger = kwiver::vital::get_logger( "scheduler.thread_per_process" );

  pipeline_t const p = pipeline();

  processes_t procs = p->get_python_processes();
  if( ! procs.empty() )
  {
    std::stringstream str;
    str << "This pipeline contains the following python processes which are not supported by this scheduler.\n";
    for ( auto proc : procs )
    {
      str << "      \"" << proc->name() << "\" of type \"" << proc->type() << "\"\n";
    }

    VITAL_THROW( incompatible_pipeline_exception, str.str());
  }

  process::names_t const names = p->process_names();

  // Scan all processes in the pipeline to see if any are not
  // compatible with this scheduler.
  for (process::name_t const& name : names)
  {
    auto proc = p->process_by_name(name);
    process::properties_t const consts = proc->properties();

    if (consts.count(process::property_no_threads))
    {
      std::string const reason =
        "The process \'" + name + "\' does not support being in its own thread.";

      VITAL_THROW( incompatible_pipeline_exception, reason);
    }
  }
}

// ------------------------------------------------------------------
thread_per_process_scheduler
::~thread_per_process_scheduler()
{
  shutdown();
}

// ------------------------------------------------------------------
void
thread_per_process_scheduler
::_start()
{
  pipeline_t const p = pipeline();
  process::names_t const names = p->process_names();

  d->m_stop_requested = false;
  d->m_error_occurred = false;
  d->m_exception_ptr = nullptr;
  d->m_error_message.clear();
  d->process_threads.clear();

  for (process::name_t const& name : names)
  {
    process_t const process = pipeline()->process_by_name(name);

    d->process_threads.emplace_back(std::bind(&priv::run_process, d.get(), process, m_logger));
  }
}

// ------------------------------------------------------------------
void
thread_per_process_scheduler
::_wait()
{
  d->join_all();

  // Re-throw any exception that occurred in a process thread
  if (d->m_error_occurred)
  {
    if (d->m_exception_ptr)
    {
      std::rethrow_exception(d->m_exception_ptr);
    }
    else
    {
      // Fallback if we only have an error message
      VITAL_THROW( incompatible_pipeline_exception, d->m_error_message );
    }
  }
}

// ------------------------------------------------------------------
void
thread_per_process_scheduler
::_pause()
{
  d->m_pause_mutex.lock();
}

// ------------------------------------------------------------------
void
thread_per_process_scheduler
::_resume()
{
  d->m_pause_mutex.unlock();
}

// ------------------------------------------------------------------
void
thread_per_process_scheduler
::_stop()
{
  d->request_stop();
}

// ============================================================================
thread_per_process_scheduler::priv
::priv()
  : process_threads()
  , m_pause_mutex()
{
}

// ------------------------------------------------------------------
thread_per_process_scheduler::priv
::~priv()
{
}

static kwiver::vital::config_block_sptr monitor_edge_config();

// ------------------------------------------------------------------
/*
 * This is the thread that runs a single process. It loops until the
 * process is complete or fails.
 */
void
thread_per_process_scheduler::priv
::run_process(process_t const& process, kwiver::vital::logger_handle_t logger)
{
  // Create the monitor edge. This is only needed for this type of scheduler.
  kwiver::vital::config_block_sptr const edge_conf = monitor_edge_config();

  name_thread(process->name());
  edge_t monitor_edge = std::make_shared<edge>(edge_conf);

  process->connect_output_port(process::port_heartbeat, monitor_edge);

  bool complete = false;

  try
  {
    while (!complete && !stop_requested() && !m_error_occurred)
    {
      // This locking will cause this thread to pause if the scheduler
      // pause() method is called.
      shared_lock_t const lock(m_pause_mutex);

      (void)lock;

      // Check if stop was requested or another thread has errored
      if (stop_requested() || m_error_occurred)
      {
        break;
      }

      process->step();

      // Check the monitor edge to see if the process is still running
      // or has completed.
      while (monitor_edge->has_data())
      {
        edge_datum_t const edat = monitor_edge->get_datum();
        datum_t const dat = edat.datum;

        // If there is a "complete" packet in the monitor edge, then the
        // process is done.
        if (dat->type() == datum::complete)
        {
          complete = true;
        }
      }
    }
  }
  catch (std::exception const& e)
  {
    // Store the exception details (thread-safe)
    std::lock_guard<std::mutex> lock(m_error_mutex);

    if (!m_error_occurred)
    {
      m_error_occurred = true;
      m_exception_ptr = std::current_exception();

      std::ostringstream msg;
      msg << "Process '" << process->name() << "' threw an exception: " << e.what();
      m_error_message = msg.str();

      LOG_ERROR(logger, m_error_message);
    }
  }
  catch (...)
  {
    // Store the exception details (thread-safe)
    std::lock_guard<std::mutex> lock(m_error_mutex);

    if (!m_error_occurred)
    {
      m_error_occurred = true;
      m_exception_ptr = std::current_exception();

      std::ostringstream msg;
      msg << "Process '" << process->name() << "' threw an unknown exception";
      m_error_message = msg.str();

      LOG_ERROR(logger, m_error_message);
    }
  }
}

// ------------------------------------------------------------------
/**
 * This function returns the config block for the "monitor_edge". The
 * monitor_edge being the one where the process generates a heart beat datum.
 *
 * Currently there is no config for these edges.
 *
 * One possibility for supplying this config would be to have it be
 * part of the scheduler config.
 */
kwiver::vital::config_block_sptr
monitor_edge_config()
{
  kwiver::vital::config_block_sptr conf = kwiver::vital::config_block::empty_config();

  // Empty config will create a default edge.

  return conf;
}

}
