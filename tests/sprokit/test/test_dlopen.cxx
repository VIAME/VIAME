/*ckwg +29
 * Copyright 2012-2013 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <test_common.h>

#include <sprokit/pipeline_util/path.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>

#if defined(_WIN32) || defined(_WIN64)
namespace
{

typedef std::basic_string<TCHAR> tstring;

}

static tstring last_windows_error();
#endif

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

#if defined(_WIN32) || defined(_WIN64)
  typedef HMODULE library_t;
#else
  typedef void* library_t;
#endif
  typedef sprokit::path_t module_path_t;

  std::string const library = argv[1];
  sprokit::path_t const path = argv[2];

  library_t handle = NULL;

#if defined(_WIN32) || defined(_WIN64)
  handle = LoadLibraryW(path.native().c_str());
#else
  handle = dlopen(path.native().c_str(), RTLD_NOW);
#endif

  if (!handle)
  {
#if defined(_WIN32) || defined(_WIN64)
    tstring const error = last_windows_error();
#else
    std::string const error = dlerror();
#endif

    TEST_ERROR("Failed to open library " << library << ": " << error);
  }
  else
  {
#if defined(_WIN32) || defined(_WIN64)
    int const ret = FreeLibrary(handle);

    if (!ret)
    {
      tstring const error = last_windows_error();
#else
    int const ret = dlclose(handle);

    if (ret)
    {
      std::string const error = dlerror();
#endif

      TEST_ERROR("Failed to close library " << library << ": " << error);
    }
  }

  return EXIT_SUCCESS;
}

#if defined(_WIN32) || defined(_WIN64)
tstring
last_windows_error()
{
  DWORD const err_code = GetLastError();
  LPTSTR str;

  DWORD const ret = FormatMessage(
    FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
    NULL,
    err_code,
    0,
    reinterpret_cast<LPTSTR>(&str),
    0,
    NULL);

  if (!ret)
  {
    TEST_ERROR("Could not get error string from system");

    exit(EXIT_FAILURE);
  }

  tstring const error = str;

  LocalFree(str);

  return error;
}
#endif
