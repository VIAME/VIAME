/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \file
 * \brief This file contains the implementation for the geo UTM coordinate
 */

#include "geo_UTM.h"

#include <iomanip>

namespace kwiver {
namespace vital {

geo_UTM
::geo_UTM()
  : zone_ (-100),
    is_north_ (true),
    easting_(0),
    northing_(0)
{ }

geo_UTM
::geo_UTM(int zone, bool north, double easting, double northing)
  : zone_ (zone),
    is_north_ (north),
    easting_ (easting),
    northing_ (northing)
{ }


bool
geo_UTM
::is_empty() const
{
  return (this->zone_ == -100);
}


bool
geo_UTM
::is_valid() const
{
  if (is_empty())
  {
    return false;
  }

  // Zone (1,60) (-1,-60)
  if ( ! ( ((zone_ >= -60) && (zone_ <= -1))
           || ((zone_ >= 1) && (zone_ <= 60)) ) )
  {
    return false;
  }

  //The magic numbers are from
  // The Universal Transverse Mercator (UTM) Grid System and Topographic Maps
  //    An Introductory Guide for Scientists & Engineers
  //    By: Joe S. Depner
  //    2011 Dec 15 Edition
  if ((easting_ < 166042) || (easting_ > 833958))
  {
    return false;
  }

  //The magic numbers are from
  // The Universal Transverse Mercator (UTM) Grid System and Topographic Maps
  //    An Introductory Guide for Scientists & Engineers
  //    By: Joe S. Depner
  //    2011 Dec 15 Edition
  if ((northing_ < 1094440) || (northing_ > 10000000))
  {
    return false;
  }

  return true;
}


geo_UTM&
geo_UTM
::set_zone(int z)
{
  zone_ = z;
  return *this;
}


geo_UTM&
geo_UTM
::set_is_north(bool v)
{
  is_north_ = v;
  return *this;
}


geo_UTM&
geo_UTM
::set_easting(double v)
{
  easting_ = v;
  return *this;
}


double
geo_UTM
::easting() const
{
  return easting_;
}


geo_UTM&
geo_UTM
::set_northing(double v)
{
  northing_ = v;
  return *this;
}


double
geo_UTM
::northing() const
{
  return northing_;
}


int geo_UTM
::zone() const
{
  return zone_;
}


bool
geo_UTM
::is_north() const
{
  return is_north_;
}


bool
geo_UTM
::operator == ( const geo_UTM &rhs ) const
{
  return ( (rhs.zone() == this->zone() )
           && (rhs.is_north() == this->is_north() )
           && (rhs.easting() == this->easting() )
           && (rhs.northing() == this->northing() ) );
}


bool
geo_UTM
::operator != ( const geo_UTM &rhs ) const
{
  return ( !( this->operator == ( rhs ) ) );
}


geo_UTM
geo_UTM
::operator=( const geo_UTM& u )
{
  if ( this != & u )
  {
    this->zone_ = u.zone_;
    this->is_north_ = u.is_north_;
    this->easting_ = u.easting_;
    this->northing_ = u.northing_;
  }

  return *this;
}


std::ostream & operator<< (std::ostream & str, const kwiver::vital::geo_UTM & obj)
{
  str << std::setprecision(16)
      << "[UTM - Z: " << obj.zone()
      << (obj.is_north() ? " (North)" : " (South)")
      << " Easting: " << obj.easting()
      << " Northing: " << obj.northing()
      << " ]";

  return str;
}

} } // end namespace
