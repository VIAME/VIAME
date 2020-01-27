/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
 * \file train_detector_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<train_detector> and train_detector
 */

#ifndef TRAIN_DETECTOR_TRAMPOLINE_TXX
#define TRAIN_DETECTOR_TRAMPOLINE_TXX


#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/train_detector.h>


template < class algorithm_def_td_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::train_detector > >
class algorithm_def_td_trampoline :
      public algorithm_trampoline<algorithm_def_td_base>
{
  public:
    using algorithm_trampoline<algorithm_def_td_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::train_detector>,
        type_name,
      );
    }
};


template< class train_detector_base=
                kwiver::vital::algo::train_detector >
class train_detector_trampoline :
      public algorithm_def_td_trampoline< train_detector_base >
{
  public:
    using algorithm_def_td_trampoline< train_detector_base>::
              algorithm_def_td_trampoline;

    void
    train_from_disk(
           kwiver::vital::category_hierarchy_sptr object_labels,
           std::vector< std::string > train_image_names,
           std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
           std::vector< std::string > test_image_names,
           std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth
         )  override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::train_detector,
        train_from_disk,
        object_labels,
        train_image_names,
        train_groundtruth,
        test_image_names,
        test_groundtruth
      );
    }

    void
    train_from_memory( kwiver::vital::category_hierarchy_sptr object_labels,
           std::vector< kwiver::vital::image_container_sptr > train_images,
           std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
           std::vector< kwiver::vital::image_container_sptr > test_images,
           std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth
         )  override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::train_detector,
        train_from_memory,
        object_labels,
        train_images,
        train_groundtruth,
        test_images,
        test_groundtruth
      );
    }
};

#endif
