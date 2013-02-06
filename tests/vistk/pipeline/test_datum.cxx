/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/datum.h>

#define TEST_ARGS ()

DECLARE_TEST(empty);
DECLARE_TEST(flush);
DECLARE_TEST(complete);
DECLARE_TEST(error);
DECLARE_TEST(new);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, empty);
  ADD_TEST(tests, flush);
  ADD_TEST(tests, complete);
  ADD_TEST(tests, error);
  ADD_TEST(tests, new);

  RUN_TEST(tests, testname);
}

IMPLEMENT_TEST(empty)
{
  vistk::datum_t const dat = vistk::datum::empty_datum();

  if (dat->type() != vistk::datum::empty)
  {
    TEST_ERROR("Datum type mismatch");
  }

  if (!dat->get_error().empty())
  {
    TEST_ERROR("An empty datum has an error string");
  }

  EXPECT_EXCEPTION(vistk::bad_datum_cast_exception,
                   dat->get_datum<int>(),
                   "retrieving a value from an empty datum");
}

IMPLEMENT_TEST(flush)
{
  vistk::datum_t const dat = vistk::datum::flush_datum();

  if (dat->type() != vistk::datum::flush)
  {
    TEST_ERROR("Datum type mismatch");
  }

  if (!dat->get_error().empty())
  {
    TEST_ERROR("A flush datum has an error string");
  }

  EXPECT_EXCEPTION(vistk::bad_datum_cast_exception,
                   dat->get_datum<int>(),
                   "retrieving a value from an flush datum");
}

IMPLEMENT_TEST(complete)
{
  vistk::datum_t const dat = vistk::datum::complete_datum();

  if (dat->type() != vistk::datum::complete)
  {
    TEST_ERROR("Datum type mismatch");
  }

  if (!dat->get_error().empty())
  {
    TEST_ERROR("A complete datum has an error string");
  }

  EXPECT_EXCEPTION(vistk::bad_datum_cast_exception,
                   dat->get_datum<int>(),
                   "retrieving a value from a complete datum");
}

IMPLEMENT_TEST(error)
{
  vistk::datum::error_t const error = vistk::datum::error_t("An error");
  vistk::datum_t const dat = vistk::datum::error_datum(error);

  if (dat->type() != vistk::datum::error)
  {
    TEST_ERROR("Datum type mismatch");
  }

  if (dat->get_error() != error)
  {
    TEST_ERROR("An error datum did not keep the message");
  }

  EXPECT_EXCEPTION(vistk::bad_datum_cast_exception,
                   dat->get_datum<int>(),
                   "retrieving a value from an error datum");
}

IMPLEMENT_TEST(new)
{
  int const datum = 100;
  vistk::datum_t const dat = vistk::datum::new_datum(100);

  if (dat->type() != vistk::datum::data)
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

  EXPECT_EXCEPTION(vistk::bad_datum_cast_exception,
                   dat->get_datum<std::string>(),
                   "retrieving an int as a string");
}
