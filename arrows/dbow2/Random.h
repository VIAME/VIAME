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

/*
 * File: Random.h
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Date: April 2010, November 2011
 * Description: manages pseudo-random numbers
 * License: see the LICENSE_DLIB.txt file
 *
 */

#pragma once
#ifndef __D_RANDOM__
#define __D_RANDOM__

#include <cstdlib>
#include <vector>

namespace DUtils {

/// Functions to generate pseudo-random numbers
class Random
{
public:
  class UnrepeatedRandomizer;

public:
  /**
   * Sets the random number seed to the current time
   */
  static void SeedRand();

  /**
   * Sets the random number seed to the current time only the first
   * time this function is called
   */
  static void SeedRandOnce();

  /**
   * Sets the given random number seed
   * @param seed
   */
  static void SeedRand(int seed);

  /**
   * Sets the given random number seed only the first time this function
   * is called
   * @param seed
   */
  static void SeedRandOnce(int seed);

  /**
   * Returns a random number in the range [0..1]
   * @return random T number in [0..1]
   */
  template <class T>
  static T RandomValue(){
    return (T)rand()/(T)RAND_MAX;
  }

  /**
   * Returns a random number in the range [min..max]
   * @param min
   * @param max
   * @return random T number in [min..max]
   */
  template <class T>
  static T RandomValue(T min, T max){
    return Random::RandomValue<T>() * (max - min) + min;
  }

  /**
   * Returns a random int in the range [min..max]
   * @param min
   * @param max
   * @return random int in [min..max]
   */
  static int RandomInt(int min, int max);

  /**
   * Returns a random number from a gaussian distribution
   * @param mean
   * @param sigma standard deviation
   */
  template <class T>
  static T RandomGaussianValue(T mean, T sigma)
  {
    // Box-Muller transformation
    T x1, x2, w, y1;

    do {
      x1 = (T)2. * RandomValue<T>() - (T)1.;
      x2 = (T)2. * RandomValue<T>() - (T)1.;
      w = x1 * x1 + x2 * x2;
    } while ( w >= (T)1. || w == (T)0. );

    w = sqrt( ((T)-2.0 * log( w ) ) / w );
    y1 = x1 * w;

    return( mean + y1 * sigma );
  }

private:

  /// If SeedRandOnce() or SeedRandOnce(int) have already been called
  static bool m_already_seeded;

};

// ---------------------------------------------------------------------------

/// Provides pseudo-random numbers with no repetitions
class Random::UnrepeatedRandomizer
{
public:

  /**
   * Creates a randomizer that returns numbers in the range [min, max]
   * @param min
   * @param max
   */
  UnrepeatedRandomizer(int min, int max);
  ~UnrepeatedRandomizer(){}

  /**
   * Copies a randomizer
   * @param rnd
   */
  UnrepeatedRandomizer(const UnrepeatedRandomizer& rnd);

  /**
   * Copies a randomizer
   * @param rnd
   */
  UnrepeatedRandomizer& operator=(const UnrepeatedRandomizer& rnd);

  /**
   * Returns a random number not given before. If all the possible values
   * were already given, the process starts again
   * @return unrepeated random number
   */
  int get();

  /**
   * Returns whether all the possible values between min and max were
   * already given. If get() is called when empty() is true, the behaviour
   * is the same than after creating the randomizer
   * @return true iff all the values were returned
   */
  inline bool empty() const { return m_values.empty(); }

  /**
   * Returns the number of values still to be returned
   * @return amount of values to return
   */
  inline unsigned int left() const { return static_cast<unsigned int>(m_values.size()); }

  /**
   * Resets the randomizer as it were just created
   */
  void reset();

protected:

  /**
   * Creates the vector with available values
   */
  void createValues();

protected:

  /// Min of range of values
  int m_min;
  /// Max of range of values
  int m_max;

  /// Available values
  std::vector<int> m_values;

};

}

#endif

