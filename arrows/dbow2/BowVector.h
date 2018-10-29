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
 * File: BowVector.h
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: bag of words vector
 * License: see the LICENSE_DBoW2.txt file
 *
 */


#ifndef __D_T_BOW_VECTOR__
#define __D_T_BOW_VECTOR__

#include <iostream>
#include <map>
#include <vector>

namespace DBoW2 {

/// Id of words
typedef unsigned int WordId;

/// Value of a word
typedef double WordValue;

/// Id of nodes in the vocabulary treee
typedef unsigned int NodeId;

/// L-norms for normalization
enum LNorm
{
  L1,
  L2
};

/// Weighting type
enum WeightingType
{
  TF_IDF,
  TF,
  IDF,
  BINARY
};

/// Scoring type
enum ScoringType
{
  L1_NORM,
  L2_NORM,
  CHI_SQUARE,
  KL,
  BHATTACHARYYA,
  DOT_PRODUCT
};

/// Vector of words to represent images
class BowVector:
  public std::map<WordId, WordValue>
{
public:

  /**
   * Constructor
   */
  BowVector(void);

  /**
   * Destructor
   */
  ~BowVector(void);

  /**
   * Adds a value to a word value existing in the vector, or creates a new
   * word with the given value
   * @param id word id to look for
   * @param v value to create the word with, or to add to existing word
   */
  void addWeight(WordId id, WordValue v);

  /**
   * Adds a word with a value to the vector only if this does not exist yet
   * @param id word id to look for
   * @param v value to give to the word if this does not exist
   */
  void addIfNotExist(WordId id, WordValue v);

  /**
   * L1-Normalizes the values in the vector
   * @param norm_type norm used
   */
  void normalize(LNorm norm_type);

  /**
   * Prints the content of the bow vector
   * @param out stream
   * @param v
   */
  friend std::ostream& operator<<(std::ostream &out, const BowVector &v);

  /**
   * Saves the bow vector as a vector in a matlab file
   * @param filename
   * @param W number of words in the vocabulary
   */
  void saveM(const std::string &filename, size_t W) const;
};

} // namespace DBoW2

#endif
