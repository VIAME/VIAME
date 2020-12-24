// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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

IMPLEMENT_TEST(equality)
{
  sprokit::datum_t const empty1 = sprokit::datum::empty_datum();
  sprokit::datum_t const empty2 = sprokit::datum::empty_datum();
  sprokit::datum_t const flush1 = sprokit::datum::flush_datum();
  sprokit::datum_t const flush2 = sprokit::datum::flush_datum();
  sprokit::datum_t const complete1 = sprokit::datum::complete_datum();
  sprokit::datum_t const complete2 = sprokit::datum::complete_datum();

  sprokit::datum::error_t const errora = sprokit::datum::error_t("An error");
  sprokit::datum::error_t const errorb = sprokit::datum::error_t("Another error");

  sprokit::datum_t const error1a = sprokit::datum::error_datum(errora);
  sprokit::datum_t const error2a = sprokit::datum::error_datum(errora);
  sprokit::datum_t const error1b = sprokit::datum::error_datum(errorb);
  sprokit::datum_t const error2b = sprokit::datum::error_datum(errorb);

  kwiver::vital::any const dummy1 = kwiver::vital::any();
  kwiver::vital::any const dummy2 = kwiver::vital::any();
  kwiver::vital::any const in_value1 = kwiver::vital::any(1);
  kwiver::vital::any const in_value2 = kwiver::vital::any(2);
  sprokit::datum_t const value_dummy1 = sprokit::datum::new_datum(dummy1);
  sprokit::datum_t const value_dummy2 = sprokit::datum::new_datum(dummy2);
  sprokit::datum_t const value1 = sprokit::datum::new_datum(in_value1);
  sprokit::datum_t const value2a = sprokit::datum::new_datum(in_value2);
  sprokit::datum_t const value2b = sprokit::datum::new_datum(in_value2);

#define test_equality(a, b, type, desc)   \
  do                                      \
  {                                       \
    if (*a != *b)                         \
    {                                     \
      TEST_ERROR("Expected a datum with " \
                 "type " type " to be "   \
                 "equal: " desc);         \
    }                                     \
  } while (false)

#define test_self_equality(a, type) \
  test_equality(a, a, type, "self comparison")

  test_self_equality(empty1, "empty");
  test_equality(empty1, empty2, "empty", "all empty data are equivalent");

  test_self_equality(flush1, "flush");
  test_equality(flush1, flush2, "flush", "all flush data are equivalent");

  test_self_equality(complete1, "complete");
  test_equality(complete1, complete2, "complete", "all complete data are equivalent");

  test_self_equality(error1a, "error");
  test_equality(error1a, error2a, "error", "all error data with the same error string are equivalent");

  test_self_equality(error1b, "error");
  test_equality(error1b, error2b, "error", "all error data with the same error string are equivalent");

  test_self_equality(value_dummy1, "data");
  test_equality(value_dummy1, value_dummy2, "data", "empty internal data");

  test_self_equality(value1, "data");
  /// \todo Is this possible?
  //test_equality(value2a, value2b, "data", "same internal data value");

#undef test_self_equality
#undef test_equality

#define test_inequality(a, b, atype, btype, desc) \
  do                                              \
  {                                               \
    if (*a == *b)                                 \
    {                                             \
      TEST_ERROR("Expected a datum with type "    \
                 atype " to be unequal to a "     \
                 "with type " btype ": " desc);   \
    }                                             \
  } while (false)

  test_inequality(empty1, flush1, "empty", "flush", "different types");
  test_inequality(empty1, complete1, "empty", "complete", "different types");
  test_inequality(empty1, error1a, "empty", "error", "different types");
  test_inequality(empty1, error1b, "empty", "error", "different types");
  test_inequality(empty1, value_dummy1, "empty", "data", "different types");
  test_inequality(empty1, value1, "empty", "data", "different types");

  test_inequality(flush1, complete1, "flush", "complete", "different types");
  test_inequality(flush1, error1a, "flush", "error", "different types");
  test_inequality(flush1, error1b, "flush", "error", "different types");
  test_inequality(flush1, value_dummy1, "flush", "data", "different types");
  test_inequality(flush1, value1, "flush", "data", "different types");

  test_inequality(complete1, error1a, "complete", "error", "different types");
  test_inequality(complete1, error1b, "complete", "error", "different types");
  test_inequality(complete1, value_dummy1, "complete", "data", "different types");
  test_inequality(complete1, value1, "complete", "data", "different types");

  test_inequality(error1a, error1b, "error", "error", "different error strings");
  test_inequality(error1a, value_dummy1, "error", "data", "different types");
  test_inequality(error1a, value1, "error", "data", "different types");

  test_inequality(value_dummy1, value1, "data", "data", "different internal data");

#undef test_inequality
}
