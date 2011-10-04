/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "plane_ref.h"

/**
 * \file plane_ref.cxx
 *
 * \brief Implementation of reference planes.
 */

namespace vistk
{

plane_ref::reference_t const plane_ref::ref_invalid = reference_t(0x00000000);
plane_ref::reference_t const plane_ref::ref_tracking_world = reference_t(0x00000001);

plane_ref
::plane_ref()
  : m_ref(ref_invalid)
{
}

plane_ref
::plane_ref(reference_t ref)
  : m_ref(ref)
{
}

plane_ref
::~plane_ref()
{
}

bool
plane_ref
::is_valid() const
{
  return (m_ref != ref_invalid);
}

plane_ref::reference_t
plane_ref
::reference() const
{
  return m_ref;
}

bool
plane_ref
::operator == (plane_ref const& ref) const
{
  return ((m_ref != ref_invalid) &&
          (m_ref == ref.m_ref));
}

}
