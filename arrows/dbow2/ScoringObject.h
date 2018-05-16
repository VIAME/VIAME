/*ckwg +29
* Copyright 2017 by Kitware, Inc.
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

/**
 * File: ScoringObject.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions to compute bow scores
 * License: see the LICENSE_DBoW2.txt file
 *
 */

#ifndef __D_T_SCORING_OBJECT__
#define __D_T_SCORING_OBJECT__

#include "BowVector.h"

namespace DBoW2 {

/// Base class of scoring functions
class GeneralScoring
{
public:
  /**
   * Computes the score between two vectors. Vectors must be sorted and
   * normalized if necessary
   * @param v (in/out)
   * @param w (in/out)
   * @return score
   */
  virtual double score(const BowVector &v, const BowVector &w) const = 0;

  /**
   * Returns whether a vector must be normalized before scoring according
   * to the scoring scheme
   * @param norm norm to use
   * @return true iff must normalize
   */
  virtual bool mustNormalize(LNorm &norm) const = 0;

  /// Log of epsilon
  static const double LOG_EPS;
  // If you change the type of WordValue, make sure you change also the
  // epsilon value (this is needed by the KL method)

  virtual ~GeneralScoring() {} //!< Required for virtual base classes
};

/**
 * Macro for defining Scoring classes
 * @param NAME name of class
 * @param MUSTNORMALIZE if vectors must be normalized to compute the score
 * @param NORM type of norm to use when MUSTNORMALIZE
 */
#define __SCORING_CLASS(NAME, MUSTNORMALIZE, NORM) \
  NAME: public GeneralScoring \
  { public: \
    /** \
     * Computes score between two vectors \
     * @param v \
     * @param w \
     * @return score between v and w \
     */ \
    virtual double score(const BowVector &v, const BowVector &w) const; \
    \
    /** \
     * Says if a vector must be normalized according to the scoring function \
     * @param norm (out) if true, norm to use
     * @return true iff vectors must be normalized \
     */ \
    virtual inline bool mustNormalize(LNorm &norm) const  \
      { norm = NORM; return MUSTNORMALIZE; } \
  }

/// L1 Scoring object
class __SCORING_CLASS(L1Scoring, true, L1);

/// L2 Scoring object
class __SCORING_CLASS(L2Scoring, true, L2);

/// Chi square Scoring object
class __SCORING_CLASS(ChiSquareScoring, true, L1);

/// KL divergence Scoring object
class __SCORING_CLASS(KLScoring, true, L1);

/// Bhattacharyya Scoring object
class __SCORING_CLASS(BhattacharyyaScoring, true, L1);

/// Dot product Scoring object
class __SCORING_CLASS(DotProductScoring, false, L1);

#undef __SCORING_CLASS

} // namespace DBoW2

#endif

