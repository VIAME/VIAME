// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

 /**
 * \file
 * \brief Source file for tv_refine_search
 */

#include "tv_refine_search.h"

#include <sstream>
#include <iomanip>
#include <limits>

#include <vil/vil_convert.h>
#include <vil/vil_save.h>
#include <vil/vil_math.h>

#include <vector>

#include <vnl/vnl_double_3.h>
#include <vnl/vnl_double_2.h>

#include <vital/logger/logger.h>

namespace kwiver {
namespace arrows {
namespace super3d {

void
refine_depth(vil_image_view<double> &cost_volume,
             const vil_image_view<double> &g,
             vil_image_view<double> &d,
             unsigned int iterations,
             double theta0,
             double theta_end,
             double lambda,
             double epsilon,
             depth_refinement_monitor *drm)
{
  static vital::logger_handle_t logger =
    vital::get_logger("arrows.super3d.refine_depth");

  vil_image_view<double> sqrt_cost_range(cost_volume.ni(), cost_volume.nj(), 1);
  double a_step = 1.0 / cost_volume.nplanes();

  for (unsigned int j = 0; j < cost_volume.nj(); j++)
  {
    for (unsigned int i = 0; i < cost_volume.ni(); i++)
    {
      double min = std::numeric_limits<double>::infinity();
      double max = -std::numeric_limits<double>::infinity();
      unsigned int min_k = 0;
      for (unsigned int k = 0; k < cost_volume.nplanes(); k++)
      {
        const double &cost = cost_volume(i,j,k);
        if (cost < min) {
          min = cost;
          min_k = k;
        }
        if (cost > max)
        {
          max = cost;
        }
      }
      sqrt_cost_range(i,j) = std::sqrt(max - min);
      d(i,j) = (min_k + 0.5) * a_step;
    }
  }

  vil_image_view<double> q(cost_volume.ni(), cost_volume.nj(), 2);
  q.fill(0.0);

  vil_image_view<double> a(cost_volume.ni(), cost_volume.nj(), 1);

  double theta = theta0;
  double denom = log(10.0);
  double orders = log(theta0)/denom - log(theta_end)/denom;
  double beta = orders / static_cast<double>(iterations);

  for (unsigned int iter = 1; iter <= iterations; iter++)
  {
    LOG_TRACE(logger, "theta: " << theta);
    min_search_bound(a, d, cost_volume, sqrt_cost_range, theta, lambda);
    huber(q, d, a, g, theta, 0.25/theta, epsilon);
    theta = pow(10.0, log(theta)/denom - beta);

    if (drm)
    {
      if (drm->callback_)
      {
        depth_refinement_monitor::update_data data;
        if (!(iter % drm->interval_))
        {
          data.current_result.deep_copy(d);
        }
        data.num_iterations = iter;
        // if the callback returns false, that means
        // the user has requested early termination
        if (!drm->callback_(data))
        {
          break;
        }
      }
    }
  }
}

//*****************************************************************************

//semi-implicit gradient ascent on q and descent on d
void huber(vil_image_view<double> &q,
           vil_image_view<double> &d,
           const vil_image_view<double> &a,
           const vil_image_view<double> &g,
           double theta,
           double step,
           double epsilon)
{
  unsigned int ni = d.ni() - 1, nj = d.nj() - 1;
  double stepsilon1 = 1.0 + step*epsilon;
#pragma omp parallel for
  for (int64_t j = 0; j < nj; j++)
  {
    for (unsigned int i = 0; i < ni; i++)
    {
      double &x = q(i,j,0), &y = q(i,j,1);
      double dij = d(i,j);
      x = (x + step * g(i,j) * (d(i+1,j) - dij))/stepsilon1;
      y = (y + step * g(i,j) * (d(i,j+1) - dij))/stepsilon1;

      //truncate vectors
      double mag = x*x + y*y;
      if (mag > 1.0f)
      {
        mag = sqrt(mag);
        x /= mag;
        y /= mag;
      }
    }
  }

  q(ni,nj,0) = q(ni-1,nj,0);
  q(ni,nj,1) = q(ni,nj-1,1);

  double theta_inv = 1.0 / theta, denom = (1.0 + (step / theta));
#pragma omp parallel for
  for (int64_t j = 0; j < d.nj(); j++)
  {
    for (unsigned int i = 0; i < d.ni(); i++)
    {
      //add scaled divergence
      double divx = q(i,j,0), divy = q(i,j,1);
      if (i > 0)  divx -=  q(i-1,j,0);
      if (j > 0)  divy -=  q(i,j-1,1);

      double &dij = d(i,j);
      dij = (dij + step * (g(i,j) * (divx + divy) + theta_inv * a(i,j)))/denom;
    }
  }
}

//*****************************************************************************

namespace {

/// Interpolate the offset to the subsampled minimum by fitting a parabola
/// This function fits a parabola to 3 points: (-1, ym1), (0, y0), (1, yp1)
/// and estimates the X location of the minimum.  It is assumed that y0 < ym1
/// and y0 < yp1.
inline
double
interp_offset(double ym1, double y0, double yp1)
{
  const double d1 = yp1 - ym1;
  const double d2 = 2 * y0 - ym1 - yp1;
  return d2 == 0.0 ? 0.0 : d1 / (2 * d2);
}

}

//*****************************************************************************

void
min_search_bound(vil_image_view<double> &a,
  const vil_image_view<double> &d,
  const vil_image_view<double> &cost_volume,
  const vil_image_view<double> &sqrt_cost_range,
  double theta,
  double lambda)
{
  const int S = static_cast<int>(cost_volume.nplanes());
  const double a_step = 1.0 / S;
  const int last_plane = S - 1;
  const double coeff = (1.0 / (2.0 * theta * lambda * S * S));
  const double range_coeff = std::sqrt(2.0 * theta * lambda);

  const std::ptrdiff_t istep_c = cost_volume.istep();
  const std::ptrdiff_t jstep_c = cost_volume.jstep();
  const std::ptrdiff_t pstep_c = cost_volume.planestep();

#pragma omp parallel for
  for (int64_t j = 0; j < d.nj(); j++)
  {
    const double* row_c = cost_volume.top_left_ptr();
    row_c += (j * jstep_c);

    const double* col_c = row_c;
    for (unsigned int i = 0; i < d.ni(); i++, col_c += istep_c)
    {
      const double sqrt_range = sqrt_cost_range(i, j);
      if (!std::isfinite(sqrt_range))
      {
        a(i, j) = d(i, j);
        continue;
      }
      const int r = std::min(last_plane,
                             std::max(0, static_cast<int>(S * range_coeff
                                                            * sqrt_range)));
      const double dij = d(i, j) * S - 0.5;
      const int init_k = std::min(last_plane, std::max(0, static_cast<int>(dij)));

      // compute the search range and clip between 0 and S-1
      // note that when dij is outside the volume range [0,1] this
      // range needs to be shifted to the closest edge of the volume
      // for example if dij < 0 then search in [0, r] and
      // if dij > 1 search in [S-1-r, 1]
      const int min_k = std::max(0, init_k - r);
      const int max_k = std::min(last_plane, init_k + r);

      int bestk = init_k;
      const double diff = dij - bestk;
      double best_e = coeff*diff*diff + *(col_c + bestk);
      const double* cost = col_c + min_k;
      for (int k = min_k; k <= max_k; ++k, cost += pstep_c)
      {
        if (k == init_k || *cost < 0.0 || *cost > best_e)
        {
          continue;
        }
        const double dd = dij - k;
        const double e = coeff * dd * dd + (*cost);
        if (e < best_e)
        {
          best_e = e;
          bestk = k;
        }
      }
      // fit a parabola to estimate the subsample offset for the best k
      if (bestk > 0 && bestk < S - 1)
      {
        cost = col_c + bestk;
        const double diff2 = 2 * coeff * (dij - static_cast<double>(bestk));
        const double ym1 = *(cost - pstep_c) + diff2 + coeff;
        const double yp1 = *(cost + pstep_c) - diff2 + coeff;
        const double offset = interp_offset(ym1, *cost, yp1);
        a(i, j) = (static_cast<double>(bestk) + offset + 0.5) * a_step;
      }
      else
      {
        a(i, j) = (static_cast<double>(bestk) + 0.5) * a_step;
      }
    }
  }
}

//*****************************************************************************

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver
