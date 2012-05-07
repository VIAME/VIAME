/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "score_mask.h"

#include <boost/make_shared.hpp>

/**
 * \file mask_scoring.cxx
 *
 * \brief Implementation of a function for scoring a mask.
 */

namespace vistk
{

scoring_result_t
score_mask(mask_t const& truth_mask, mask_t const& computed_mask)
{
  size_t const ni = truth_mask.ni();
  size_t const nj = truth_mask.nj();
  size_t const np = truth_mask.nplanes();

  ptrdiff_t const tsi = truth_mask.istep();
  ptrdiff_t const tsj = truth_mask.jstep();
  ptrdiff_t const tsp = truth_mask.planestep();

  ptrdiff_t const csi = computed_mask.istep();
  ptrdiff_t const csj = computed_mask.jstep();
  ptrdiff_t const csp = computed_mask.planestep();

  typedef mask_t::pixel_type pixel_t;

  pixel_t const* to = truth_mask.top_left_ptr();
  pixel_t const* co = computed_mask.top_left_ptr();

  scoring_result::count_t hit = 0;
  scoring_result::count_t miss = 0;
  scoring_result::count_t truth = 0;

  for (size_t i = 0; i < ni; ++i, to += tsi, co += csi)
  {
    pixel_t const* tr = to;
    pixel_t const* cr = co;

    for (size_t j = 0; j < nj; ++j, tr += tsj, cr += csj)
    {
      pixel_t const* tp = tr;
      pixel_t const* cp = cr;

      for (size_t p = 0; p < np; ++p, tp += tsp, cp += csp)
      {
        pixel_t const& t = *tp;
        pixel_t const& c = *cp;

        if (t)
        {
          ++truth;
        }

        if (c)
        {
          if (t)
          {
            ++hit;
          }
          else
          {
            ++miss;
          }
        }
      }
    }
  }

  return boost::make_shared<scoring_result>(hit, miss, truth);
}

}
