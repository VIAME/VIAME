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
 * File: Random.cpp
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Date: April 2010
 * Description: manages pseudo-random numbers
 * License: see the LICENSE_DLIB.txt file
 *
 */

#include "Random.h"
#include "Timestamp.h"
#include <cstdlib>
using namespace std;

bool DUtils::Random::m_already_seeded = false;

void DUtils::Random::SeedRand(){
  Timestamp time;
  time.setToCurrentTime();
  srand((unsigned)time.getFloatTime());
}

void DUtils::Random::SeedRandOnce()
{
  if(!m_already_seeded)
  {
    DUtils::Random::SeedRand();
    m_already_seeded = true;
  }
}

void DUtils::Random::SeedRand(int seed)
{
  srand(seed);
}

void DUtils::Random::SeedRandOnce(int seed)
{
  if(!m_already_seeded)
  {
    DUtils::Random::SeedRand(seed);
    m_already_seeded = true;
  }
}

int DUtils::Random::RandomInt(int min, int max){
  int d = max - min + 1;
  return int(((double)rand()/((double)RAND_MAX + 1.0)) * d) + min;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer(int min, int max)
{
  if(min <= max)
  {
    m_min = min;
    m_max = max;
  }
  else
  {
    m_min = max;
    m_max = min;
  }

  createValues();
}

// ---------------------------------------------------------------------------

DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer
  (const DUtils::Random::UnrepeatedRandomizer& rnd)
{
  *this = rnd;
}

// ---------------------------------------------------------------------------

int DUtils::Random::UnrepeatedRandomizer::get()
{
  if(empty()) createValues();

  DUtils::Random::SeedRandOnce();

  int k = DUtils::Random::RandomInt(0, static_cast<int>(m_values.size()-1));
  int ret = m_values[k];
  m_values[k] = m_values.back();
  m_values.pop_back();

  return ret;
}

// ---------------------------------------------------------------------------

void DUtils::Random::UnrepeatedRandomizer::createValues()
{
  int n = m_max - m_min + 1;

  m_values.resize(n);
  for(int i = 0; i < n; ++i) m_values[i] = m_min + i;
}

// ---------------------------------------------------------------------------

void DUtils::Random::UnrepeatedRandomizer::reset()
{
  if((int)m_values.size() != m_max - m_min + 1) createValues();
}

// ---------------------------------------------------------------------------

DUtils::Random::UnrepeatedRandomizer&
DUtils::Random::UnrepeatedRandomizer::operator=
  (const DUtils::Random::UnrepeatedRandomizer& rnd)
{
  if(this != &rnd)
  {
    this->m_min = rnd.m_min;
    this->m_max = rnd.m_max;
    this->m_values = rnd.m_values;
  }
  return *this;
}

// ---------------------------------------------------------------------------


