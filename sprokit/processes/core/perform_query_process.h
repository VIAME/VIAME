// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_PERFORM_QUERY_PROCESS_H_
#define _KWIVER_PERFORM_QUERY_PROCESS_H_

#include "kwiver_processes_export.h"

#include <sprokit/pipeline/process.h>

#include <memory>

namespace kwiver
{

// -----------------------------------------------------------------------------
/**
 * \class perform_query_process
 *
 * \brief Runs a database query and produces an output result
 *
 * \iports
 * \iport{database_query}
 * \iport{iqr_feedback}
 *
 * \oports
 * \oport{query_result}
 */
class KWIVER_PROCESSES_NO_EXPORT perform_query_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "perform_query",
               "Perform a query." )

  perform_query_process( vital::config_block_sptr const& config );
  virtual ~perform_query_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class perform_query_process

} // end namespace

#endif /* _KWIVER_PERFORM_QUERY_PROCESS_H_ */
