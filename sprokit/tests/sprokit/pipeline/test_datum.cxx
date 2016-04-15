/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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

#include <sprokit/pipeline/datum.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

IMPLEMENT_TEST(empty)
{
  sprokit::datum_t const dat = sprokit::datum::empty_datum();

  if (dat->type() != sprokit::datum::empty)
  {
    TEST_ERROR("Datum type mismatch");
  }

  if (!dat->get_error().empty())
  {
    TEST_ERROR("An empty datum has an error string");
  }

  EXPECT_EXCEPTION(sprokit::bad_datum_cast_exception,
                   dat->get_datum<int>(),
                   "retrieving a value from an empty datum");
}

IMPLEMENT_TEST(flush)
{
  sprokit::datum_t const dat = sprokit::datum::flush_datum();

  if (dat->type() != sprokit::datum::flush)
  {
    TEST_ERROR("Datum type mismatch");
  }

  if (!dat->get_error().empty())
  {
    TEST_ERROR("A flush datum has an error string");
  }

  EXPECT_EXCEPTION(sprokit::bad_datum_cast_exception,
                   dat->get_datum<int>(),
                   "retrieving a value from an flush datum");
}

IMPLEMENT_TEST(complete)
{
  sprokit::datum_t const dat = sprokit::datum::complete_datum();

  if (dat->type() != sprokit::datum::complete)
  {
    TEST_ERROR("Datum type mismatch");
  }

  if (!dat->get_error().empty())
  {
    TEST_ERROR("A complete datum has an error string");
  }

  EXPECT_EXCEPTION(sprokit::bad_datum_cast_exception,
                   dat->get_datum<int>(),
                   "retrieving a value from a complete datum");
}

IMPLEMENT_TEST(error)
{
  sprokit::datum::error_t const error = sprokit::datum::error_t("An error");
  sprokit::datum_t const dat = sprokit::datum::error_datum(error);

  if (dat->type() != sprokit::datum::error)
  {
    TEST_ERROR("Datum type mismatch");
  }

  if (dat->get_error() != error)
  {
    TEST_ERROR("An error datum did not keep the message");
  }

  EXPECT_EXCEPTION(sprokit::bad_datum_cast_exception,
                   dat->get_datum<int>(),
                   "retrieving a value from an error datum");
}

IMPLEMENT_TEST(new)
{
  int const datum = 100;
  sprokit::datum_t const dat = sprokit::datum::new_datum(100);

  if (dat->type() != sprokit::datum::data)
  {
    TEST_ERROR("Datum type mismatch");
  }

  if (!dat->get_error().empty())
  {
    TEST_ERROR("A new datum has an error string");
  }

  int const get_datum = dat->get_datum<int>();

  if (datum != get_datum)
  {
    TEST_ERROR("Did not get same value out as put into datum");
  }

  EXPECT_EXCEPTION(sprokit::bad_datum_cast_exception,
                   dat->get_datum<std::string>(),
                   "retrieving an int as a string");
}
