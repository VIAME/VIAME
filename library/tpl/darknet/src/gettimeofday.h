/* gettimeofday() for MSVC.
 *
 * darknet's sources include "gettimeofday.h" and `utils.c` calls the
 * function. Upstream ships this header for the Windows build; the vendoring
 * that brought darknet into this tree took the sources that reference it and
 * not the header itself, so it is supplied here under the name they use.
 *
 * `struct timeval` comes from <winsock2.h> on Windows, which is also why the
 * lean-and-mean guard matters: several darknet sources include <winsock2.h>
 * themselves, and letting <windows.h> drag in the original <winsock.h> first
 * redefines it.
 */

#ifndef DARKNET_GETTIMEOFDAY_H
#define DARKNET_GETTIMEOFDAY_H

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>   /* struct timeval */
#include <windows.h>

/* FILETIME counts 100ns ticks from 1601-01-01; the Unix epoch is this far
 * along it. */
#define DARKNET_EPOCH_DELTA_100NS 116444736000000000ULL

struct timezone;

static __inline int
gettimeofday( struct timeval* tv, struct timezone* tz )
{
  FILETIME filetime;
  ULARGE_INTEGER ticks;

  ( void ) tz;   /* obsolete in POSIX too, and darknet passes NULL */

  if( !tv )
  {
    return -1;
  }

  /* The precise form: GetSystemTimeAsFileTime is milliseconds-granular on
   * older Windows, and darknet uses this to time training steps. */
  GetSystemTimePreciseAsFileTime( &filetime );

  ticks.LowPart = filetime.dwLowDateTime;
  ticks.HighPart = filetime.dwHighDateTime;
  ticks.QuadPart -= DARKNET_EPOCH_DELTA_100NS;

  tv->tv_sec = ( long ) ( ticks.QuadPart / 10000000ULL );
  tv->tv_usec = ( long ) ( ( ticks.QuadPart % 10000000ULL ) / 10ULL );

  return 0;
}

#else
#include <sys/time.h>
#endif /* _WIN32 */

#endif /* DARKNET_GETTIMEOFDAY_H */
