// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for match matrix computation
 */

#ifndef KWIVER_ARROWS_CORE_MATCH_MATRIX_H_
#define KWIVER_ARROWS_CORE_MATCH_MATRIX_H_

#include <vital/vital_config.h>
#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/types/track_set.h>
#include <Eigen/Sparse>

#include <map>

namespace kwiver {
namespace arrows {

/// Compute the match matrix from a track set
/**
 *  This function computes an NxN integer symmetric matrix such that matrix
 *  element (i,j) is the number of feature tracks with corresponding points
 *  on both frames i and j.  The diagonal (i,i) is the number of features
 *  on frame i.  The frame ids corresponding to each row/column are returned
 *  in a vector.
 *
 *  \param[in]     tracks  The tracks from which to extract the match matrix
 *  \param[in,out] frames  The vector of frame ids used in the match matrix.
 *                         If empty, this will be filled in will all available
 *                         frame ids in the track set.
 *  \return an NxN symmetric match matrix
 */
KWIVER_ALGO_CORE_EXPORT
Eigen::SparseMatrix<unsigned int>
match_matrix(vital::track_set_sptr tracks,
             std::vector<vital::frame_id_t>& frames);

/// Compute a score for each track based on its importance to the match matrix.
/**
 * Using the match matrix (as computed by vital::match_matrix) assign a score
 * to each track that is proportional to that tracks importance in reproducing
 * the matrix.  That is, the top N scoring tracks should provide the best
 * approximation to the coverage of match matrix if only those N tracks are
 * used.  Tracks are scored as the sum of one over the mm(i,j) where mm(i,j)
 * is the match matrix entry at (i,j) for every frame i and j in the track.
 *
 */
KWIVER_ALGO_CORE_EXPORT
std::map<vital::track_id_t, double>
match_matrix_track_importance(vital::track_set_sptr tracks,
                              std::vector<vital::frame_id_t> const& frames,
                              Eigen::SparseMatrix<unsigned int> const& mm);
} // end namespace arrows
} // end namespace kwiver

#endif
