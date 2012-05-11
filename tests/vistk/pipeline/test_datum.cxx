/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/datum.h>

#include <exception>
#include <iostream>
#include <string>

#include <cstdlib>

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return EXIT_FAILURE;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_empty();
static void test_flush();
static void test_complete();
static void test_error();
static void test_new();

void
run_test(std::string const& test_name)
{
  if (test_name == "empty")
  {
    test_empty();
  }
  else if (test_name == "flush")
  {
    test_flush();
  }
  else if (test_name == "complete")
  {
    test_complete();
  }
  else if (test_name == "error")
  {
    test_error();
  }
  else if (test_name == "new")
  {
    test_new();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

void
test_empty()
{
  vistk::datum_t dat = vistk::datum::empty_datum();

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

void
test_flush()
{
  vistk::datum_t dat = vistk::datum::flush_datum();

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

void
test_complete()
{
  vistk::datum_t dat = vistk::datum::complete_datum();

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

void
test_error()
{
  vistk::datum::error_t const error = vistk::datum::error_t("An error");
  vistk::datum_t dat = vistk::datum::error_datum(error);

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

void
test_new()
{
  int const datum = 100;
  vistk::datum_t dat = vistk::datum::new_datum(100);

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
