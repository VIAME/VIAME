/* Enough of pthreads for darknet, on Windows.
 *
 * MSVC has no <pthread.h> and darknet reaches for one in two headers. The
 * whole surface it uses is six names, so this maps them rather than adding
 * a pthreads-for-Windows dependency to the tree:
 *
 *   pthread_t                  HANDLE
 *   pthread_create             _beginthreadex, through a trampoline
 *   pthread_join               WaitForSingleObject
 *   pthread_mutex_t            SRWLOCK
 *   PTHREAD_MUTEX_INITIALIZER  SRWLOCK_INIT
 *   pthread_mutex_lock/unlock  Acquire/ReleaseSRWLockExclusive
 *
 * SRWLOCK rather than CRITICAL_SECTION because darknet initialises its
 * mutexes statically with PTHREAD_MUTEX_INITIALIZER, which a
 * CRITICAL_SECTION cannot do -- it needs an InitializeCriticalSection call.
 * An SRWLOCK is also non-recursive, which is what a default pthread mutex
 * is, so nothing that deadlocks here would have worked there either.
 *
 * `_beginthreadex` and not `CreateThread`: darknet's threads call into the
 * C runtime, and CreateThread leaves the per-thread CRT state uninitialised.
 */

#ifndef DARKNET_PTHREAD_WIN32_H
#define DARKNET_PTHREAD_WIN32_H

#ifdef _WIN32

/* Before <windows.h>, or it pulls in the original <winsock.h> and every
 * darknet source that later includes <winsock2.h> -- darkunistd.h and
 * http_stream.cpp do -- redefines fd_set, hostent, linger and WSAData. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <process.h>
#include <stdlib.h>

typedef HANDLE pthread_t;
typedef SRWLOCK pthread_mutex_t;

#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT

/* The start routine darknet passes is `void *(*)(void *)`; _beginthreadex
 * wants `unsigned (__stdcall *)(void *)`. The pair is carried on the heap
 * because the caller's stack frame may be gone before the thread runs. */
typedef struct darknet_thread_start_
{
  void* ( *routine )( void* );
  void* argument;
} darknet_thread_start_;

static unsigned __stdcall
darknet_thread_trampoline_( void* raw )
{
  darknet_thread_start_ start = *( darknet_thread_start_* ) raw;
  free( raw );
  start.routine( start.argument );
  return 0;
}

/* Returns 0 on success, as pthread_create does -- darknet tests the return
 * value truthily, so a Windows-style nonzero-is-success would invert it. */
static __inline int
pthread_create( pthread_t* thread, void* attr,
                void* ( *routine )( void* ), void* argument )
{
  darknet_thread_start_* start;

  ( void ) attr;

  start = ( darknet_thread_start_* ) malloc( sizeof( *start ) );
  if( !start )
  {
    return -1;
  }

  start->routine = routine;
  start->argument = argument;

  *thread = ( HANDLE ) _beginthreadex( NULL, 0, darknet_thread_trampoline_,
                                       start, 0, NULL );
  if( !*thread )
  {
    free( start );
    return -1;
  }

  return 0;
}

static __inline int
pthread_join( pthread_t thread, void** value )
{
  /* Nothing in darknet reads a thread's return value, and carrying one back
   * would mean keeping the trampoline's storage alive past the join. */
  if( value )
  {
    *value = NULL;
  }

  if( WaitForSingleObject( thread, INFINITE ) != WAIT_OBJECT_0 )
  {
    return -1;
  }

  CloseHandle( thread );
  return 0;
}

static __inline int
pthread_mutex_init( pthread_mutex_t* mutex, void* attr )
{
  ( void ) attr;
  InitializeSRWLock( mutex );
  return 0;
}

static __inline int
pthread_mutex_destroy( pthread_mutex_t* mutex )
{
  /* An SRWLOCK holds no resources to release. */
  ( void ) mutex;
  return 0;
}

static __inline int
pthread_mutex_lock( pthread_mutex_t* mutex )
{
  AcquireSRWLockExclusive( mutex );
  return 0;
}

static __inline int
pthread_mutex_unlock( pthread_mutex_t* mutex )
{
  ReleaseSRWLockExclusive( mutex );
  return 0;
}

#endif /* _WIN32 */

#endif /* DARKNET_PTHREAD_WIN32_H */
