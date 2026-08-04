/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "train_supervisor.h"

#include <kwiversys/SystemTools.hxx>

#include <plugins/core/utilities_file.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace viame {
namespace claude {

const char* const child_env_marker = "VIAME_LLM_TRAIN_CHILD";

namespace {

// ---------------------------------------------------------------------------
// Small shared helpers (compiled on all platforms)

std::string trim_copy( const std::string& s )
{
  size_t b = s.find_first_not_of( " \t\r\n" );
  if( b == std::string::npos )
  {
    return "";
  }
  size_t e = s.find_last_not_of( " \t\r\n" );
  return s.substr( b, e - b + 1 );
}

bool iequals_prefix( const std::string& line, const std::string& prefix )
{
  if( line.size() < prefix.size() )
  {
    return false;
  }
  for( size_t i = 0; i < prefix.size(); ++i )
  {
    if( std::toupper( static_cast< unsigned char >( line[i] ) ) !=
        std::toupper( static_cast< unsigned char >( prefix[i] ) ) )
    {
      return false;
    }
  }
  return true;
}

std::string timestamp_now()
{
  std::time_t t = std::time( nullptr );
  char buf[ 32 ];
  std::strftime( buf, sizeof( buf ), "%Y-%m-%d %H:%M:%S", std::localtime( &t ) );
  return buf;
}

// A single suggestion / decision returned by claude
struct llm_decision
{
  std::string action; // "continue", "restart", or "abort" ("" if unspecified)
  std::string config; // alternate training config file, if suggested
  std::vector< std::pair< std::string, std::string > > settings;
  std::vector< std::string > reasons;

  bool empty() const
  {
    return action.empty() && config.empty() && settings.empty();
  }
};

// Parse the strict line protocol requested from claude:
//   ACTION: continue|restart|abort
//   CONFIG: <file>
//   SETTING: key=value
//   REASON: <text>
llm_decision parse_llm_response( const std::string& text )
{
  llm_decision out;
  std::istringstream ss( text );
  std::string raw;

  while( std::getline( ss, raw ) )
  {
    std::string line = trim_copy( raw );

    if( iequals_prefix( line, "ACTION:" ) )
    {
      std::string v = trim_copy( line.substr( 7 ) );
      std::transform( v.begin(), v.end(), v.begin(),
        []( unsigned char c ){ return std::tolower( c ); } );

      if( v == "continue" || v == "restart" || v == "abort" )
      {
        out.action = v;
      }
    }
    else if( iequals_prefix( line, "CONFIG:" ) )
    {
      out.config = trim_copy( line.substr( 7 ) );
    }
    else if( iequals_prefix( line, "SETTING:" ) )
    {
      std::string kv = trim_copy( line.substr( 8 ) );
      size_t eq = kv.find( '=' );

      if( eq != std::string::npos && eq > 0 )
      {
        out.settings.emplace_back( trim_copy( kv.substr( 0, eq ) ),
                                   trim_copy( kv.substr( eq + 1 ) ) );
      }
    }
    else if( iequals_prefix( line, "REASON:" ) )
    {
      out.reasons.push_back( trim_copy( line.substr( 7 ) ) );
    }
  }
  return out;
}

} // end anonymous namespace

// ===========================================================================
#ifndef _WIN32

namespace {

std::atomic< int > g_interrupt_signal( 0 );

void interrupt_handler( int sig )
{
  g_interrupt_signal = sig;
}

std::string sh_quote( const std::string& s )
{
  std::string out = "'";
  for( char c : s )
  {
    if( c == '\'' )
    {
      out += "'\\''";
    }
    else
    {
      out += c;
    }
  }
  out += "'";
  return out;
}

// Run a shell command and capture its stdout (stderr discarded)
std::string capture_command( const std::string& cmd, size_t max_bytes = 16384 )
{
  std::string full = cmd + " 2>/dev/null";
  FILE* pipe = popen( full.c_str(), "r" );

  if( !pipe )
  {
    return "";
  }

  std::string out;
  char buf[ 4096 ];
  size_t n;

  while( ( n = fread( buf, 1, sizeof( buf ), pipe ) ) > 0 )
  {
    if( out.size() < max_bytes )
    {
      out.append( buf, std::min( n, max_bytes - out.size() ) );
    }
  }

  pclose( pipe );
  return out;
}

std::string read_file_capped( const std::string& path, size_t max_bytes,
                              bool from_end = false )
{
  std::ifstream in( path, std::ios::binary );

  if( !in )
  {
    return "";
  }

  in.seekg( 0, std::ios::end );
  std::streamoff size = in.tellg();
  std::streamoff start = 0;

  if( from_end && size > static_cast< std::streamoff >( max_bytes ) )
  {
    start = size - static_cast< std::streamoff >( max_bytes );
  }

  in.seekg( start );
  std::string out;
  out.resize( std::min< std::streamoff >( size - start,
    static_cast< std::streamoff >( max_bytes ) ) );
  in.read( &out[0], out.size() );
  out.resize( in.gcount() );
  return out;
}

// One-shot snapshot of machine utilization for the prompt context
std::string gather_resource_snapshot()
{
  std::ostringstream out;
  out << "[" << timestamp_now() << "]" << std::endl;

  std::ifstream loadavg( "/proc/loadavg" );
  std::string l1, l5, l15;
  if( loadavg >> l1 >> l5 >> l15 )
  {
    out << "cpu load (1/5/15m): " << l1 << " " << l5 << " " << l15
        << "  cores: " << std::thread::hardware_concurrency() << std::endl;
  }

  std::ifstream meminfo( "/proc/meminfo" );
  std::string line;
  while( std::getline( meminfo, line ) )
  {
    if( line.find( "MemTotal" ) == 0 || line.find( "MemAvailable" ) == 0 )
    {
      out << line << std::endl;
    }
  }

  std::string gpus = capture_command(
    "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total"
    " --format=csv,noheader" );

  if( !gpus.empty() )
  {
    out << "gpu (index, name, util, mem used, mem total):" << std::endl << gpus;
  }
  else
  {
    out << "no NVIDIA GPU detected via nvidia-smi" << std::endl;
  }

  return out.str();
}

// ---------------------------------------------------------------------------
// The supervisor itself

class llm_supervisor
{
public:
  explicit llm_supervisor( const llm_train_options& options )
    : o( options )
  {
  }

  int run();

private:
  const llm_train_options& o;

  std::string log_path_;
  std::string status_path_;
  std::string settings_path_;
  std::ofstream status_;
  std::ofstream log_stream_;

  // Applied setting over-rides (user settings file first, then claude's)
  std::vector< std::pair< std::string, std::string > > settings_;
  std::string active_config_;

  std::deque< std::string > resource_history_;
  int consult_id_ = 0;
  int restart_count_ = 0;
  bool claude_usable_ = true;

  // Child process state
  pid_t child_pid_ = -1;
  int child_fd_ = -1;
  std::thread reader_;
  mutable std::mutex tail_mutex_;
  std::string log_tail_;

  static const size_t max_tail_bytes = 262144; // keep last 256 KB in memory
  static const size_t prompt_log_bytes = 32768; // log context given to claude

  void note( const std::string& msg )
  {
    if( status_.is_open() )
    {
      status_ << timestamp_now() << " " << msg << std::endl;
    }
    std::cout << "[llm-supervisor] " << msg << std::endl;
  }

  bool preflight();
  bool consult( const std::string& prompt, std::string& response );
  void push_resource_snapshot();
  std::string resource_history_text() const;

  void merge_settings( const std::vector< std::pair< std::string, std::string > >& in );
  void write_settings_file();
  void load_user_settings_file();

  bool suggestion_phase();
  std::string build_common_context() const;
  std::string build_suggest_prompt() const;
  std::string build_monitor_prompt() const;
  std::string build_failure_prompt( int exit_code ) const;

  std::vector< std::string > build_child_argv() const;
  bool launch_child();
  void finish_reader();
  void stop_child();
  std::string tail_copy() const;

  int monitor_loop();
};

// ---------------------------------------------------------------------------
bool
llm_supervisor
::preflight()
{
  std::string cmd = sh_quote( o.claude_cmd ) + " --version > /dev/null 2>&1";
  return std::system( cmd.c_str() ) == 0;
}

// ---------------------------------------------------------------------------
// Run one claude consultation. Returns false only when claude could not be run,
// which is distinct from it running and having nothing to say.
bool
llm_supervisor
::consult( const std::string& prompt, std::string& response )
{
  response.clear();
  consult_id_++;

  std::string id = std::to_string( consult_id_ );
  std::string prompt_path = append_path( o.state_dir, "prompt_" + id + ".txt" );
  std::string resp_path = append_path( o.state_dir, "response_" + id + ".txt" );
  std::string err_path = append_path( o.state_dir, "stderr_" + id + ".txt" );

  {
    std::ofstream pf( prompt_path );
    pf << prompt;
  }

  std::ostringstream cmd;

  // Bound each consultation; a hung claude call must not stall monitoring
  if( std::system( "command -v timeout > /dev/null 2>&1" ) == 0 )
  {
    cmd << "timeout 600 ";
  }

  cmd << sh_quote( o.claude_cmd ) << " -p";

  if( !o.model.empty() )
  {
    cmd << " --model " << sh_quote( o.model );
  }

  cmd << " < " << sh_quote( prompt_path )
      << " > " << sh_quote( resp_path )
      << " 2> " << sh_quote( err_path );

  note( "consulting claude (" + std::to_string( prompt.size() ) + " byte prompt)" );

  int rc = std::system( cmd.str().c_str() );

  if( rc != 0 )
  {
    std::string err = trim_copy( read_file_capped( err_path, 2048, true ) );
    note( "claude consultation failed (rc=" + std::to_string( rc ) +
          ( err.empty() ? std::string( ")" ) : "): " + err ) );
    return false;
  }

  response = read_file_capped( resp_path, 65536 );
  return true;
}

// ---------------------------------------------------------------------------
void
llm_supervisor
::push_resource_snapshot()
{
  resource_history_.push_back( gather_resource_snapshot() );

  while( resource_history_.size() > 12 )
  {
    resource_history_.pop_front();
  }
}

std::string
llm_supervisor
::resource_history_text() const
{
  std::string out;
  for( const auto& s : resource_history_ )
  {
    out += s + "\n";
  }
  return out;
}

// ---------------------------------------------------------------------------
void
llm_supervisor
::merge_settings( const std::vector< std::pair< std::string, std::string > >& in )
{
  for( const auto& kv : in )
  {
    bool replaced = false;

    for( auto& existing : settings_ )
    {
      if( existing.first == kv.first )
      {
        existing.second = kv.second;
        replaced = true;
        break;
      }
    }

    if( !replaced )
    {
      settings_.push_back( kv );
    }
  }
}

void
llm_supervisor
::write_settings_file()
{
  std::ofstream out( settings_path_ );
  out << "# Generated by viame train --llm-assist" << std::endl;

  for( const auto& kv : settings_ )
  {
    out << kv.first << "=" << kv.second << std::endl;
  }
}

void
llm_supervisor
::load_user_settings_file()
{
  if( o.user_settings_file.empty() )
  {
    return;
  }

  std::ifstream in( o.user_settings_file );
  std::string line;
  std::vector< std::pair< std::string, std::string > > parsed;

  while( std::getline( in, line ) )
  {
    line = trim_copy( line );

    if( line.empty() || line[0] == '#' )
    {
      continue;
    }

    size_t eq = line.find( '=' );

    if( eq != std::string::npos && eq > 0 )
    {
      parsed.emplace_back( trim_copy( line.substr( 0, eq ) ),
                           trim_copy( line.substr( eq + 1 ) ) );
    }
  }

  merge_settings( parsed );
}

// ---------------------------------------------------------------------------
std::string
llm_supervisor
::build_common_context() const
{
  std::ostringstream out;

  out << "Original command line:" << std::endl
      << "  " << o.original_cmdline << std::endl << std::endl;

  if( !active_config_.empty() )
  {
    out << "Active training config file: " << active_config_ << std::endl;
  }
  else if( !o.original_detector.empty() )
  {
    out << "Training detector type: " << o.original_detector << std::endl;
  }

  if( !settings_.empty() )
  {
    out << "Setting over-rides currently applied:" << std::endl;
    for( const auto& kv : settings_ )
    {
      out << "  " << kv.first << "=" << kv.second << std::endl;
    }
  }

  out << std::endl
      << "=== Effective training configuration (key=value dump, may be truncated) ==="
      << std::endl << o.config_text << std::endl;

  if( !o.dataset_summary.empty() )
  {
    out << "=== Dataset ===" << std::endl << o.dataset_summary << std::endl;
  }

  return out.str();
}

std::string
llm_supervisor
::build_suggest_prompt() const
{
  std::ostringstream out;

  out <<
    "You are an expert supervising an object detector training run for VIAME\n"
    "(Video and Image Analytics for Multiple Environments). Training has not\n"
    "started yet. Review the setup and suggest improvements.\n\n";

  out << build_common_context();

  out << "=== Machine resources ===" << std::endl
      << gather_resource_snapshot() << std::endl;

  if( !o.trainable_types.empty() )
  {
    out << "=== Trainable detector types in this install ===" << std::endl
        << o.trainable_types << std::endl;
  }

  if( !o.available_configs.empty() )
  {
    out << "=== Packaged training config files (configs/pipelines) ===" << std::endl
        << o.available_configs << std::endl;
  }

  out <<
    "Suggest, if warranted:\n"
    "1. A better-suited packaged training config (different model architecture\n"
    "   or input resolution) for this dataset and hardware, via CONFIG.\n"
    "2. Hyperparameter / configuration over-rides via SETTING, using EXACT keys\n"
    "   from the effective configuration dump above.\n"
    "3. The user wants training to maximize this machine's resources: GPU memory\n"
    "   close to full without out-of-memory risk, and CPU data-loading workers\n"
    "   matched to the core count. Adjust batch size / worker style settings\n"
    "   accordingly if the keys exist in the configuration.\n\n"
    "Rules:\n"
    "- Only suggest changes you are confident improve this run; suggesting\n"
    "  nothing is a valid answer.\n"
    "- Do not override values the user set explicitly on the command line.\n"
    "- Do not invent configuration keys.\n\n"
    "Respond with ONLY these line formats (no prose, no markdown):\n"
    "CONFIG: <one filename from the packaged config list>   (optional, at most one)\n"
    "SETTING: <config:key>=<value>                          (zero or more)\n"
    "REASON: <one short line per suggestion>                (zero or more)\n";

  return out.str();
}

std::string
llm_supervisor
::build_monitor_prompt() const
{
  std::ostringstream out;

  out <<
    "You are monitoring a running VIAME object detector training process.\n"
    "Restarting the process loses training progress, so only recommend restart\n"
    "when something is genuinely wrong (crash loop imminent, deadlock/stall,\n"
    "NaN losses, out-of-memory) or when the run just started and is severely\n"
    "underutilizing the machine and better settings would fix it.\n\n";

  out << "Restarts so far: " << restart_count_ << " (maximum "
      << o.max_restarts << ")" << std::endl << std::endl;

  out << build_common_context();

  out << "=== Machine resource snapshots (oldest to newest) ===" << std::endl
      << resource_history_text() << std::endl;

  out << "=== Most recent training output ===" << std::endl
      << tail_copy() << std::endl;

  out <<
    "\nAssess the run and respond with ONLY these line formats:\n"
    "ACTION: continue | restart | abort\n"
    "SETTING: <config:key>=<value>   (only with restart: settings to change)\n"
    "REASON: <one short line>\n";

  return out.str();
}

std::string
llm_supervisor
::build_failure_prompt( int exit_code ) const
{
  std::ostringstream out;

  out <<
    "A VIAME object detector training process you are supervising has EXITED\n"
    "with a failure. Decide whether to restart it with revised settings, or\n"
    "abort if the failure cannot be fixed by changing settings.\n\n";

  out << "Exit code: " << exit_code << std::endl;
  out << "Restarts so far: " << restart_count_ << " (maximum "
      << o.max_restarts << ")" << std::endl << std::endl;

  out << build_common_context();

  out << "=== Machine resource snapshots (oldest to newest) ===" << std::endl
      << resource_history_text() << std::endl;

  out << "=== Final training output before exit ===" << std::endl
      << tail_copy() << std::endl;

  out <<
    "\nCommon fixable causes: CUDA out of memory (reduce batch size or input\n"
    "resolution), bad learning rate (NaN loss), too many data workers, or a\n"
    "transient I/O error (restart unchanged).\n"
    "\nRespond with ONLY these line formats:\n"
    "ACTION: restart | abort\n"
    "SETTING: <config:key>=<value>   (settings to change before restart)\n"
    "REASON: <one short line>\n";

  return out.str();
}

// ---------------------------------------------------------------------------
std::vector< std::string >
llm_supervisor
::build_child_argv() const
{
  std::vector< std::string > argv;

  // Resolve our own executable for relaunch
  char exe_buf[ 4096 ];
  ssize_t len = ::readlink( "/proc/self/exe", exe_buf, sizeof( exe_buf ) - 1 );

  if( len > 0 )
  {
    argv.push_back( std::string( exe_buf, len ) );
  }
  else
  {
    argv.push_back( "viame" ); // fall back to PATH lookup via execvp
  }

  argv.insert( argv.end(), o.child_args_base.begin(), o.child_args_base.end() );

  if( !active_config_.empty() )
  {
    argv.push_back( "-c" );
    argv.push_back( active_config_ );
  }
  else if( !o.original_detector.empty() )
  {
    argv.push_back( "-d" );
    argv.push_back( o.original_detector );
  }

  if( !settings_.empty() )
  {
    argv.push_back( "--settings-file" );
    argv.push_back( settings_path_ );
  }

  return argv;
}

bool
llm_supervisor
::launch_child()
{
  std::vector< std::string > argv = build_child_argv();

  {
    std::ostringstream msg;
    msg << "launching training process:";
    for( const auto& a : argv )
    {
      msg << " " << a;
    }
    note( msg.str() );
  }

  int fds[ 2 ];

  if( ::pipe( fds ) != 0 )
  {
    note( "failed to create pipe for training process" );
    return false;
  }

  pid_t pid = ::fork();

  if( pid < 0 )
  {
    ::close( fds[0] );
    ::close( fds[1] );
    note( "failed to fork training process" );
    return false;
  }

  if( pid == 0 )
  {
    // Child: own process group so the whole training tree can be signaled
    ::setpgid( 0, 0 );
    ::dup2( fds[1], STDOUT_FILENO );
    ::dup2( fds[1], STDERR_FILENO );
    ::close( fds[0] );
    ::close( fds[1] );

    ::setenv( child_env_marker, "1", 1 );

    std::vector< char* > cargv;
    for( auto& a : argv )
    {
      cargv.push_back( const_cast< char* >( a.c_str() ) );
    }
    cargv.push_back( nullptr );

    ::execvp( cargv[0], cargv.data() );
    std::perror( "viame train llm-supervisor exec" );
    ::_exit( 127 );
  }

  // Parent
  ::setpgid( pid, pid ); // also set here to avoid a signal/exec race
  ::close( fds[1] );

  child_pid_ = pid;
  child_fd_ = fds[0];

  {
    std::lock_guard< std::mutex > lock( tail_mutex_ );
    log_tail_.clear();
  }

  log_stream_ << std::endl << "===== training process started (pid "
              << pid << ") at " << timestamp_now() << " =====" << std::endl;

  reader_ = std::thread( [this]()
  {
    char buf[ 4096 ];
    ssize_t n;

    while( ( n = ::read( child_fd_, buf, sizeof( buf ) ) ) > 0 )
    {
      fwrite( buf, 1, n, stdout );
      fflush( stdout );

      std::lock_guard< std::mutex > lock( tail_mutex_ );
      log_stream_.write( buf, n );
      log_stream_.flush();
      log_tail_.append( buf, n );

      if( log_tail_.size() > max_tail_bytes )
      {
        size_t cut = log_tail_.size() - ( max_tail_bytes / 2 );
        size_t nl = log_tail_.find( '\n', cut );
        log_tail_.erase( 0, nl == std::string::npos ? cut : nl + 1 );
      }
    }
  } );

  return true;
}

void
llm_supervisor
::finish_reader()
{
  if( reader_.joinable() )
  {
    reader_.join();
  }
  if( child_fd_ >= 0 )
  {
    ::close( child_fd_ );
    child_fd_ = -1;
  }
}

void
llm_supervisor
::stop_child()
{
  if( child_pid_ <= 0 )
  {
    return;
  }

  // Escalate: SIGINT lets trainers spool down checkpoints, then TERM, then KILL
  const std::vector< std::pair< int, int > > plan =
    { { SIGINT, 60 }, { SIGTERM, 20 }, { SIGKILL, 10 } };

  for( const auto& step : plan )
  {
    ::kill( -child_pid_, step.first );

    for( int i = 0; i < step.second; ++i )
    {
      int status = 0;
      if( ::waitpid( child_pid_, &status, WNOHANG ) == child_pid_ )
      {
        // Reap any stragglers still holding the output pipe open so the
        // reader thread sees EOF and can be joined
        ::kill( -child_pid_, SIGKILL );
        finish_reader();
        child_pid_ = -1;
        return;
      }
      std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
    }
  }

  ::waitpid( child_pid_, nullptr, 0 );
  finish_reader();
  child_pid_ = -1;
}

std::string
llm_supervisor
::tail_copy() const
{
  std::lock_guard< std::mutex > lock( tail_mutex_ );

  if( log_tail_.size() <= prompt_log_bytes )
  {
    return log_tail_;
  }
  return log_tail_.substr( log_tail_.size() - prompt_log_bytes );
}

// ---------------------------------------------------------------------------
bool
llm_supervisor
::suggestion_phase()
{
  std::string response;

  if( !consult( build_suggest_prompt(), response ) )
  {
    claude_usable_ = false;
    return false;
  }

  llm_decision d = parse_llm_response( response );

  if( d.config.empty() && d.settings.empty() )
  {
    note( "claude suggested no configuration changes" );
    return true;
  }

  std::cout << std::endl << "LLM suggested training changes:" << std::endl;

  std::string new_config;

  if( !d.config.empty() )
  {
    // Resolve relative suggestions against the packaged config directory
    new_config = d.config;

    if( !does_file_exist( new_config ) )
    {
      const char* viame_install = std::getenv( "VIAME_INSTALL" );

      if( viame_install )
      {
        std::string candidate = append_path(
          append_path( std::string( viame_install ), "configs/pipelines" ),
          d.config );

        new_config = does_file_exist( candidate ) ? candidate : "";
      }
      else
      {
        new_config = "";
      }
    }

    if( new_config.empty() )
    {
      note( "ignoring suggested config (not found): " + d.config );
    }
    else
    {
      std::cout << "  config: " << new_config << std::endl;
    }
  }

  for( const auto& kv : d.settings )
  {
    std::cout << "  setting: " << kv.first << "=" << kv.second << std::endl;
  }
  for( const auto& r : d.reasons )
  {
    std::cout << "  reason: " << r << std::endl;
  }

  bool apply = true;

  if( !o.no_query )
  {
    std::cout << std::endl << "Apply these suggestions? (y/n) ";
    std::string reply;
    std::cin >> reply;
    apply = ( reply == "y" || reply == "Y" || reply == "yes" || reply == "Yes" );
  }

  if( apply )
  {
    if( !new_config.empty() )
    {
      active_config_ = new_config;
    }
    merge_settings( d.settings );
    note( "applied claude startup suggestions" );
  }
  else
  {
    note( "user declined claude startup suggestions" );
  }

  return true;
}

// ---------------------------------------------------------------------------
int
llm_supervisor
::monitor_loop()
{
  if( !launch_child() )
  {
    return EXIT_FAILURE;
  }

  auto last_consult = std::chrono::steady_clock::now();

  while( true )
  {
    if( g_interrupt_signal )
    {
      note( "interrupt received, stopping training process" );
      stop_child();
      return 128 + g_interrupt_signal;
    }

    int status = 0;
    pid_t r = ::waitpid( child_pid_, &status, WNOHANG );

    if( r == child_pid_ )
    {
      finish_reader();
      child_pid_ = -1;

      int rc = WIFEXITED( status ) ? WEXITSTATUS( status )
             : WIFSIGNALED( status ) ? 128 + WTERMSIG( status ) : EXIT_FAILURE;

      if( rc == 0 )
      {
        note( "training process finished successfully" );
        return EXIT_SUCCESS;
      }

      note( "training process exited with code " + std::to_string( rc ) );

      if( restart_count_ >= o.max_restarts )
      {
        note( "maximum restart count reached, giving up" );
        return rc;
      }

      push_resource_snapshot();

      llm_decision d;

      if( claude_usable_ )
      {
        std::string response;

        if( consult( build_failure_prompt( rc ), response ) )
        {
          d = parse_llm_response( response );
        }
      }

      if( d.action == "abort" )
      {
        note( "claude recommends aborting: " +
              ( d.reasons.empty() ? std::string( "no reason given" )
                                  : d.reasons[0] ) );
        return rc;
      }

      // Restart (also the fallback when claude is unavailable or unclear)
      for( const auto& reason : d.reasons )
      {
        note( "restart reason: " + reason );
      }

      merge_settings( d.settings );
      write_settings_file();
      restart_count_++;

      note( "restarting training (attempt " + std::to_string( restart_count_ ) +
            " of " + std::to_string( o.max_restarts ) + ")" );

      if( !launch_child() )
      {
        return EXIT_FAILURE;
      }

      last_consult = std::chrono::steady_clock::now();
      continue;
    }

    std::this_thread::sleep_for( std::chrono::seconds( 1 ) );

    auto now = std::chrono::steady_clock::now();

    if( claude_usable_ && o.poll_seconds > 0 &&
        std::chrono::duration_cast< std::chrono::seconds >(
          now - last_consult ).count() >= o.poll_seconds )
    {
      push_resource_snapshot();

      std::string response;
      bool consulted = consult( build_monitor_prompt(), response );
      last_consult = std::chrono::steady_clock::now();

      if( !consulted )
      {
        continue; // keep the run going and try again at the next checkup
      }

      llm_decision d = parse_llm_response( response );

      if( d.action == "abort" )
      {
        note( "claude recommends aborting: " +
              ( d.reasons.empty() ? std::string( "no reason given" )
                                  : d.reasons[0] ) );
        stop_child();
        return EXIT_FAILURE;
      }
      else if( d.action == "restart" )
      {
        if( restart_count_ >= o.max_restarts )
        {
          note( "claude recommended restart, but maximum restart count reached;"
                " continuing current run" );
        }
        else
        {
          for( const auto& reason : d.reasons )
          {
            note( "restart reason: " + reason );
          }

          stop_child();
          merge_settings( d.settings );
          write_settings_file();
          restart_count_++;

          note( "restarting training (attempt " +
                std::to_string( restart_count_ ) + " of " +
                std::to_string( o.max_restarts ) + ")" );

          if( !launch_child() )
          {
            return EXIT_FAILURE;
          }
        }
      }
      else
      {
        note( "claude checkup: continue" +
              ( d.reasons.empty() ? std::string() : " - " + d.reasons[0] ) );
      }
    }
  }
}

// ---------------------------------------------------------------------------
int
llm_supervisor
::run()
{
  create_folder( o.state_dir );

  log_path_ = append_path( o.state_dir, "training_output.log" );
  status_path_ = append_path( o.state_dir, "supervisor_status.log" );
  settings_path_ = append_path( o.state_dir, "suggested_settings.txt" );

  status_.open( status_path_, std::ios::app );
  log_stream_.open( log_path_, std::ios::app );

  if( !preflight() )
  {
    if( o.required )
    {
      std::cerr << "--llm-assist on: claude executable found but not runnable ("
                << o.claude_cmd << " --version failed)" << std::endl;
      return EXIT_FAILURE;
    }
    return -1; // auto mode: silently fall back to normal training
  }

  note( "LLM-assisted training enabled using " + o.claude_cmd );

  active_config_ = o.original_config;
  load_user_settings_file();

  if( !suggestion_phase() && !claude_usable_ )
  {
    if( o.required )
    {
      note( "claude unusable for suggestions; continuing supervised with"
            " automatic restart on failure only" );
    }
    else
    {
      note( "claude unusable; falling back to normal training" );
      return -1;
    }
  }

  if( !settings_.empty() )
  {
    write_settings_file();
  }

  push_resource_snapshot();

  // Install interrupt forwarding for the duration of the monitored run
  struct sigaction sa;
  std::memset( &sa, 0, sizeof( sa ) );
  sa.sa_handler = interrupt_handler;
  sigaction( SIGINT, &sa, nullptr );
  sigaction( SIGTERM, &sa, nullptr );

  return monitor_loop();
}

} // end anonymous namespace

// ===========================================================================
std::string
find_claude_binary( const std::string& cmd_override )
{
  std::string cmd = cmd_override.empty() ? "claude" : cmd_override;

  // Explicit paths are checked directly, bare names searched on PATH
  if( cmd.find( '/' ) != std::string::npos )
  {
    return does_file_exist( cmd ) ? cmd : "";
  }

  return kwiversys::SystemTools::FindProgram( cmd );
}

// ---------------------------------------------------------------------------
int
run_llm_supervised_training( const llm_train_options& options )
{
  llm_supervisor supervisor( options );
  return supervisor.run();
}

#else // _WIN32

std::string
find_claude_binary( const std::string& )
{
  return "";
}

int
run_llm_supervised_training( const llm_train_options& options )
{
  if( options.required )
  {
    std::cerr << "--llm-assist is not supported on Windows" << std::endl;
    return EXIT_FAILURE;
  }
  return -1;
}

#endif

} // namespace claude
} // namespace viame
