/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "utils.h"

#ifdef HAVE_SETPROCTITLE
#include <cstdlib>
#elif defined(__linux__)
#include <sys/prctl.h>
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#if defined(_WIN32) || defined(_WIN64)
static void SetThreadName(DWORD dwThreadID, char* threadName);
#endif

namespace vistk
{

bool name_thread(thread_name_t const& name)
{
#ifdef HAVE_SETPROCTITLE
  setproctitle("%s", name.c_str());
#elif defined(__linux__)
  int const ret = prctl(PR_SET_NAME, reinterpret_cast<unsigned long>(name.c_str()), 0, 0, 0);

  return (ret < 0);
#elif defined(_WIN32) || defined(_WIN64)
#ifndef NDEBUG
  SetThreadName(-1, const_cast<char*>(name.c_str()));
#else
  return false;
#endif
#else
  return false;
#endif

  return true;
}

}

#if defined(_WIN32) || defined(_WIN64)

// Code obtained from http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx
static DWORD const MS_VC_EXCEPTION = 0x406D1388;

#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
   DWORD dwType; // Must be 0x1000.
   LPCSTR szName; // Pointer to name (in user addr space).
   DWORD dwThreadID; // Thread ID (-1 = caller thread).
   DWORD dwFlags; // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

void SetThreadName(DWORD dwThreadID, char* threadName)
{
   THREADNAME_INFO info;
   info.dwType = 0x1000;
   info.szName = threadName;
   info.dwThreadID = dwThreadID;
   info.dwFlags = 0;

   __try
   {
      RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
   }
   __except(EXCEPTION_EXECUTE_HANDLER)
   {
   }
}

#endif
