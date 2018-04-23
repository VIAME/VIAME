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
 * File: QueryResults.h
 * Date: March, November 2011
 * Author: Dorian Galvez-Lopez
 * Description: structure to store results of database queries
 * License: see the LICENSE_DBoW2.txt file
 *
 */

#ifndef __D_T_QUERY_RESULTS__
#define __D_T_QUERY_RESULTS__

#include <vector>

namespace DBoW2 {

/// Id of entries of the database
typedef unsigned int EntryId;

/// Single result of a query
class Result
{
public:

  /// Entry id
  EntryId Id;

  /// Score obtained
  double Score;

  /// debug
  int nWords; // words in common
  // !!! this is filled only by Bhatt score!
  // (and for BCMatching, BCThresholding then)

  double bhatScore, chiScore;
  /// debug

  // only done by ChiSq and BCThresholding
  double sumCommonVi;
  double sumCommonWi;
  double expectedChiScore;
  /// debug

  /**
   * Empty constructors
   */
  inline Result(){}

  /**
   * Creates a result with the given data
   * @param _id entry id
   * @param _score score
   */
  inline Result(EntryId _id, double _score): Id(_id), Score(_score){}

  /**
   * Compares the scores of two results
   * @return true iff this.score < r.score
   */
  inline bool operator<(const Result &r) const
  {
    return this->Score < r.Score;
  }

  /**
   * Compares the scores of two results
   * @return true iff this.score > r.score
   */
  inline bool operator>(const Result &r) const
  {
    return this->Score > r.Score;
  }

  /**
   * Compares the entry id of the result
   * @return true iff this.id == id
   */
  inline bool operator==(EntryId id) const
  {
    return this->Id == id;
  }

  /**
   * Compares the score of this entry with a given one
   * @param s score to compare with
   * @return true iff this score < s
   */
  inline bool operator<(double s) const
  {
    return this->Score < s;
  }

  /**
   * Compares the score of this entry with a given one
   * @param s score to compare with
   * @return true iff this score > s
   */
  inline bool operator>(double s) const
  {
    return this->Score > s;
  }

  /**
   * Compares the score of two results
   * @param a
   * @param b
   * @return true iff a.Score > b.Score
   */
  static inline bool gt(const Result &a, const Result &b)
  {
    return a.Score > b.Score;
  }

  /**
   * Compares the scores of two results
   * @return true iff a.Score > b.Score
   */
  inline static bool ge(const Result &a, const Result &b)
  {
    return a.Score > b.Score;
  }

  /**
   * Returns true iff a.Score >= b.Score
   * @param a
   * @param b
   * @return true iff a.Score >= b.Score
   */
  static inline bool geq(const Result &a, const Result &b)
  {
    return a.Score >= b.Score;
  }

  /**
   * Returns true iff a.Score >= s
   * @param a
   * @param s
   * @return true iff a.Score >= s
   */
  static inline bool geqv(const Result &a, double s)
  {
    return a.Score >= s;
  }


  /**
   * Returns true iff a.Id < b.Id
   * @param a
   * @param b
   * @return true iff a.Id < b.Id
   */
  static inline bool ltId(const Result &a, const Result &b)
  {
    return a.Id < b.Id;
  }

  /**
   * Prints a string version of the result
   * @param os ostream
   * @param ret Result to print
   */
  friend std::ostream & operator<<(std::ostream& os, const Result& ret );
};

/// Multiple results from a query
class QueryResults: public std::vector<Result>
{
public:

  /**
   * Multiplies all the scores in the vector by factor
   * @param factor
   */
  inline void scaleScores(double factor);

  /**
   * Prints a string version of the results
   * @param os ostream
   * @param ret QueryResults to print
   */
  friend std::ostream & operator<<(std::ostream& os, const QueryResults& ret );

  /**
   * Saves a matlab file with the results
   * @param filename
   */
  void saveM(const std::string &filename) const;

};

// --------------------------------------------------------------------------

inline void QueryResults::scaleScores(double factor)
{
  for(QueryResults::iterator qit = begin(); qit != end(); ++qit)
    qit->Score *= factor;
}

// --------------------------------------------------------------------------

} // namespace TemplatedBoW

#endif

