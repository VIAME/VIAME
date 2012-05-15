/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCORING_SCORING_RESULT_H
#define VISTK_SCORING_SCORING_RESULT_H

#include "scoring-config.h"

#include <boost/cstdint.hpp>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>

namespace vistk
{

/**
 * \class scoring_result scoring_result.h <vistk/scoring/scoring_result.h>
 *
 * \brief A class which represents the result of a scoring operation.
 */
class VISTK_SCORING_EXPORT scoring_result
{
  public:
    /// A type for the counts of categories when scoring.
    typedef uint64_t count_t;
    /// A type for the result of a derived score.
    typedef double result_t;

    /**
     * \brief Constructor.
     *
     * \param hit The number of computed result that match the truth.
     * \param miss The number of computed result that do not match the truth.
     * \param truth The number of truth instances.
     */
    scoring_result(count_t hit, count_t miss, count_t truth);
    /**
     * \brief Constructor.
     *
     * \param hit The number of computed result that match the truth.
     * \param miss The number of computed result that do not match the truth.
     * \param truth The number of truth instances.
     * \param possible The number of possible detections.
     */
    scoring_result(count_t hit, count_t miss, count_t truth, count_t possible);
    /**
     * \brief Destructor.
     */
    ~scoring_result();

    /**
     * \brief
     *
     * \returns The percentage of the truth that was detected.
     */
    result_t percent_detection() const;
    /**
     * \brief
     *
     * \returns The precision of the results.
     */
    result_t precision() const;
    /**
     * \brief
     *
     * \returns The specificity of the results.
     */
    result_t specificity() const;

    /// The number of computed result that match the truth.
    count_t const hit_count;
    /// The number of computed result that do not match the truth.
    count_t const miss_count;
    /// The total number of truth instances.
    count_t const truth_count;
    /// The total number of possible detections.
    boost::optional<count_t> const possible_count;
};

/// A handle to a scoring result.
typedef boost::shared_ptr<scoring_result const> scoring_result_t;

/// A collection of scoring results.
typedef std::vector<scoring_result_t> scoring_results_t;

/**
 * \brief An addition operator for \ref scoring_result.
 *
 * \param lhs The left hand side of the operation.
 * \param rhs The right hand side of the operation.
 */
scoring_result_t VISTK_SCORING_EXPORT operator + (scoring_result_t const& lhs, scoring_result_t const& rhs);

}

#endif // VISTK_SCORING_SCORING_RESULT_H
