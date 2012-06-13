/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_UTM_H
#define VISTK_UTILITIES_UTM_H

#include "utilities-config.h"

#include <boost/operators.hpp>

/**
 * \file utm.h
 *
 * \brief UTM data structures.
 */

namespace vistk
{

/**
 * \class utm_zone_t utm.h <vistk/utilities/utm.h>
 *
 * \brief A zone in the UTM coordinate space.
 */
class VISTK_UTILITIES_EXPORT utm_zone_t
  : boost::equality_comparable<vistk::utm_zone_t>
{
  public:
    /// The type for the UTM zone.
    typedef int zone_t;

    /**
     * \enum hemisphere_t
     *
     * \brief The hemisphere within the zone.
     */
    typedef enum
    {
      /// The northern hemisphere.
      hemi_north,
      /// The southern hemisphere.
      hemi_south,

      /// The default hemisphere.
      hemi_default = hemi_north
    } hemisphere_t;

    /**
     * \brief Constructor.
     */
    utm_zone_t();
    /**
     * \brief Constructor.
     *
     * \param z The zone.
     * \param h The hemisphere within the zone.
     */
    utm_zone_t(zone_t z, hemisphere_t h);
    /**
     * \brief Destructor.
     */
    ~utm_zone_t();

    /**
     * \brief Query for the zone.
     *
     * \returns The zone.
     */
    zone_t zone() const;
    /**
     * \brief Query for the hemisphere.
     *
     * \returns The hemisphere of the zone.
     */
    hemisphere_t hemisphere() const;

    /**
     * \brief Set the utm zone.
     *
     * \param z The zone.
     */
    void set_zone(zone_t z);
    /**
     * \brief Set the hemisphere of the zone.
     *
     * \param h The hemisphere.
     */
    void set_hemisphere(hemisphere_t h);

    /**
     * \brief Equality operator for utm zones.
     *
     * \param utm The zone to compare to.
     *
     * \returns True if \p utm and \c this are equivalent, false otherwise.
     */
    bool operator == (utm_zone_t const& utm) const;

    /// The invalid zone.
    static zone_t const zone_invalid;
  private:
    zone_t m_zone;
    hemisphere_t m_hemisphere;
};

}

#endif // VISTK_UTILITIES_UTM_H
