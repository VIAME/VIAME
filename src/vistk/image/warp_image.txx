/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "warp_image.h"

#include <vil/vil_bilin_interp.h>

#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_homg_point_2d.h>
#include <vgl/vgl_intersection.h>
#include <vgl/vgl_point_2d.h>

#include <vnl/vnl_double_3.h>
#include <vnl/vnl_inverse.h>

#include <limits>

#include <cmath>

/**
 * \file warp_image.txx
 *
 * \brief Implementation of the function to warp images.
 */

namespace vistk
{

template <typename PixType>
class warp_image<PixType>::priv
{
  public:
    priv(size_t dest_width, size_t dest_height, size_t dest_planes, fill_t const& fill_value);
    ~priv();

    void clear_mask();

    image_t dest;
    mask_t mask;
    fill_t fill;
};

template <typename PixType>
warp_image<PixType>
::warp_image(size_t dest_width, size_t dest_height, size_t dest_planes, fill_t const& fill_value)
  : d(new priv(dest_width, dest_height, dest_planes, fill_value))
{
}

template <typename PixType>
warp_image<PixType>
::~warp_image()
{
}

template <typename PixType>
void
warp_image<PixType>
::clear_mask()
{
  d->clear_mask();
}

template <typename PixType>
typename warp_image<PixType>::mask_t
warp_image<PixType>
::mask() const
{
  return d->mask;
}

static bool is_identity(vistk::homography_base::transform_t const& transform);
template <typename T, typename U>
static T safe_cast(U const& value);

template <typename PixType>
typename warp_image<PixType>::image_t
warp_image<PixType>
::operator () (image_t const& image, transform_t const& transform) const
{
  size_t const sni = image.ni();
  size_t const snj = image.nj();
  size_t const snp = image.nplanes();

  if (!d->dest)
  {
    d->dest = image_t(sni, snj, 1, snp);
    d->mask = mask_t(sni, snj);
  }

  size_t const dni = d->dest.ni();
  size_t const dnj = d->dest.nj();
  size_t const dnp = d->dest.nplanes();

  if (snp != dnp)
  {
    /// \todo Throw an exception.
  }

  if (is_identity(transform))
  {
    d->dest = image;
    d->mask.fill(false);

    return d->dest;
  }

  typedef vgl_homg_point_2d<double> homog_point_t;
  typedef vgl_point_2d<double> point_t;
  typedef vnl_matrix_fixed<double, 3, 3> matrix_t;
  typedef vnl_double_3 column_t;
  typedef std::vector<column_t> columns_t;
  typedef vgl_box_2d<double> box_t;

  matrix_t const homog = transform.get_matrix();

  transform_t const inv_transform = vnl_inverse(homog);

  box_t mapped_bbox;

  size_t const snc = sni - 1;
  size_t const snr = snj - 1;

  homog_point_t const corner[4] =
    { homog_point_t(0, 0)
    , homog_point_t(snc, 0)
    , homog_point_t(snc, snr)
    , homog_point_t(0, snr)
    };

  for (size_t i = 0; i < 4; ++i)
  {
    point_t const p = inv_transform * corner[i];
    mapped_bbox.add(p);
  }

  box_t const dest_bounds(0, dni - 1, 0, dnj - 1);
  box_t const intersection = vgl_intersection(mapped_bbox, dest_bounds);

  size_t const begin_i = static_cast<size_t>(std::floor(intersection.min_x()));
  size_t const begin_j = static_cast<size_t>(std::floor(intersection.min_y()));
  size_t const end_i = static_cast<size_t>(std::floor(intersection.max_x()));
  size_t const end_j = static_cast<size_t>(std::floor(intersection.max_y()));

  ptrdiff_t const dis = d->dest.istep();
  ptrdiff_t const djs = d->dest.jstep();
  ptrdiff_t const dps = d->dest.planestep();
  ptrdiff_t const sis = image.istep();
  ptrdiff_t const sjs = image.jstep();
  ptrdiff_t const sps = image.planestep();

  size_t const factor_size = end_i - begin_i;

  column_t const homog_col_1 = homog.get_column(0);
  column_t const homog_col_2 = homog.get_column(1);
  column_t const homog_col_3 = homog.get_column(2);

  columns_t col_factors(factor_size);

  for (size_t i = 0; i < factor_size; ++i)
  {
    double const col = i + begin_i;
    col_factors[i] = col * homog_col_1 + homog_col_3;
  }

  pixel_t const* begin_src = image.top_left_ptr();
  pixel_t const* end_src = begin_src + (dnp * dps);

  pixel_t* row_start = d->dest.top_left_ptr();
  row_start += (begin_i * dis);
  row_start += (begin_j * djs);

  for (size_t j = begin_j; j < end_j; ++j, row_start += djs)
  {
    pixel_t* dest_col = row_start;

    column_t const row_factor = homog_col_2 * double(j);

    column_t* col_factor_ptr = &col_factors[0];

    for (size_t i = begin_i; i < end_i; ++i, ++col_factor_ptr, dest_col += dis)
    {
      column_t pt = row_factor + *col_factor_ptr;

      double& x = pt[0];
      double& y = pt[1];
      double const& w = pt[2];

      x /= w;
      y /= w;

      if (!((x < 0) || (y < 0) || (snc < x) || (snr < y)))
      {
        pixel_t const* sp = begin_src;

        for (pixel_t* dp = dest_col; sp < end_src; sp += sps, dp += dps)
        {
          double const v = vil_bilin_interp_raw(x, y, sp, sis, sjs);
          *dp = safe_cast<pixel_t>(v);
        }

        d->mask(i, j) = false;
      }
    }
  }

  return d->dest;
}

template <typename PixType>
warp_image<PixType>::priv
::priv(size_t dest_width, size_t dest_height, size_t dest_planes, fill_t const& fill_value)
  : dest(dest_width, dest_height, dest_planes)
  , mask(dest_width, dest_height)
  , fill(fill_value)
{
  if (fill)
  {
    dest.fill(*fill);
  }

  clear_mask();
}

template <typename PixType>
warp_image<PixType>::priv
::~priv()
{
}

template <typename PixType>
void
warp_image<PixType>::priv
::clear_mask()
{
  mask.fill(true);
}

template <typename T>
static bool fuzzy_cmp(T const& a, T const& b, T const& epsilon = std::numeric_limits<T>::epsilon());

bool
is_identity(vistk::homography_base::transform_t const& transform)
{
  typedef vnl_matrix_fixed<double, 3, 3> matrix_t;

  matrix_t const mat = transform.get_matrix();

  return (fuzzy_cmp(mat(0, 1), 0.0) && fuzzy_cmp(mat(0, 2), 0.0) &&
          fuzzy_cmp(mat(1, 0), 0.0) && fuzzy_cmp(mat(1, 2), 0.0) &&
          fuzzy_cmp(mat(2, 0), 0.0) && fuzzy_cmp(mat(2, 1), 0.0) &&
          fuzzy_cmp(mat(0, 0), mat(1, 1)) && fuzzy_cmp(mat(1, 1), mat(2, 2)));
}

template <>
VISTK_UNUSED
bool
safe_cast<bool, float>(float const& value)
{
  return fuzzy_cmp(value, float(0));
}

template <>
VISTK_UNUSED
bool
safe_cast<bool, double>(double const& value)
{
  return fuzzy_cmp(value, double(0));
}

template <typename T, typename U>
VISTK_UNUSED
T
safe_cast(U const& value)
{
  return T(value);
}

template <typename T>
bool
fuzzy_cmp(T const& a, T const& b, T const& epsilon)
{
  T const diff = std::fabs(a - b);
  return (diff <= epsilon);
}

}
