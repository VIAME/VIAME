// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#if !defined vital_simple_stats_HH_
#define vital_simple_stats_HH_

#include <limits>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <algorithm>

namespace kwiver {
namespace vital {

/// Simple statistics class
/**
 *
 * This class collects simple statistics about a set of data values.
 * The data values are passed one at a time and summary values are
 * kept to determine the minimum value, the maximum value, the average
 * value, and the standard deviation.
 */
class simple_stats
{
public:
  /**
   * \brief Constructor.
   *
   * This constructor creates a new object that is in the Reset state.
   */
  simple_stats() { reset(); }

  /**
   * \brief Reset to initial state.
   *
   * This method resets all internal accumulators to the initial state
   * and makes this object ready to accept a new data set.
   */
  void
  reset()
  {
    m_min = std::numeric_limits< double >::max();
    m_max = std::numeric_limits< double >::min();
    m_sum = 0.0;
    m_sum_sqr = 0.0;
    m_count = 0;
  }

  /**
   * \brief Add value to data set.
   *
   * This method specifies a new data value to be included in the
   * statistical soup.
   *
   * @param[in] data - data value to include.
   */
  void
  add_datum( double data )
  {
    m_min = std::min( m_min, data );
    m_max = std::max( m_max, data );
    m_count++;
    m_sum += data;
    m_sum_sqr += ( data * data );
  }

  /**
   * \brief Return average.
   *
   * This method returns the average of all the data values supplied.
   * If no data items have been added, the average is considered zero.
   *
   * @return The average of the data values.
   */
  double
  get_average() const
  {
    double avg = 0.0;
    if ( m_count > 0 )
    {
      avg = m_sum / m_count;
    }
    return ( avg );
  }

  /**
   * \brief Return standard deviation.
   *
   * This method returns the standard deviation of the data set.  If
   * no data items have been added, the standard deviation is
   * considered to be zero.
   *
   * @return The standard deviation of the data set.
   */
  double
  get_standard_deviation() const
  {
    double std_dev = 0.0;
    if ( m_count > 0 )
    {
      double average = m_sum / m_count;
      std_dev = sqrt( ( m_sum_sqr / m_count ) - ( average * average ) );
    }
    return ( std_dev );
  }

  /// @{
  /**
   * \brief Return elements from the cumulative statistics.
   *
   * This method returns the specific value from the cumulative
   * statistics being calculated.  These values are undefined if the
   * count == 0 (except count, which will be 0).
   */
  uint64_t get_count() const { return m_count; }
  double get_sum() const { return m_sum; }
  double get_min() const { return m_min; }
  double get_max() const { return m_max; }
  /// @}

  /**
   * \brief Display current statistics.
   *
   * This method displays the current accumulated staistics to the
   * specified stream.
   *
   * @param str - stream to accept output
   */
  std::ostream&
  to_stream( std::ostream& str ) const
  {
    double average = 0.0;
    double std_dev = 0.0;
    double min = 0.0;
    double max = 0.0;

    if ( m_count > 0 )
    {
      min = m_min;
      max = m_max;
      average = m_sum / m_count;
      std_dev = sqrt( ( m_sum_sqr / m_count ) - ( average * average ) );
    }

    str << std::showpoint
        << "Summary statistics\n"
        << "    Min: " << min
        << "    Max: " << max
        << "    Num samples: " << m_count
        << std::endl

        << "    Average: " << average
        << "    Std dev: " << std_dev
        << std::endl;

    return ( str );
  }

private:
  double m_min;
  double m_max;
  double m_sum;
  double m_sum_sqr;
  uint64_t m_count;
};

/**
 * \brief Output operator.
*
* This is the standard output operator for the simple_stats
* class.
*
* @param[in] str - stream to format on
* @param[in] obj - object to format
* @returns The same output stream passed in
*/

inline std::ostream & operator<< (std::ostream & str, const simple_stats & stat)
{ return stat.to_stream (str); }

// ----------------------------------------------------------------------------
/// Summary statistics.
/**
 * This class accumulates statistics and prints them to std::cout when
 * the object is destroyed. This is useful when an object of this type
 * is declared static. Then the statistics are printed when the
 * program terminates.
 */
class summary_stats
  : public simple_stats
{
public:
  summary_stats() = default;
  summary_stats( const char* descr )
    : m_descr( descr )
  {}

  ~summary_stats()
  {
    std::cout << m_descr << std::endl
              << *this;
  }

private:
  std::string m_descr;
};

} } // end namespace

#endif
