/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \file
 * \brief Implementation of refine detections using SVM
 */

#include "refine_detections_with_svm.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <deque>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <vital/vital_config.h>
#include <vital/logger/logger.h>
#include <vital/exceptions/io.h>

#include <kwiversys/SystemTools.hxx>



using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace svm{

typedef kwiversys::SystemTools ST;

/// Private implementation class
class refine_detections_with_svm::priv
{
 public:

  /// Constructor
  priv()
      : model_dir( "" ),
        id( 0 )
  {
  }

  /// Destructor
  ~priv()
  {
  }

  /// Parameters
  std::string model_dir;

  /// Variables
  unsigned id;
};


/// Constructor
refine_detections_with_svm
::refine_detections_with_svm()
    : d_( new priv() )
{
}


/// Destructor
refine_detections_with_svm
::~refine_detections_with_svm()
{
}


/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
refine_detections_with_svm
::get_configuration() const
{
  vital::config_block_sptr config = vital::algo::refine_detections::get_configuration();

  config->set_value( "model_dir", d_->model_dir,
                     "The directory where the SVM models are placed." );

  return config;
}


/// Set this algorithm's properties via a config block
void
refine_detections_with_svm
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d_->model_dir = config->get_value<std::string>( "model_dir" );
}

/// Check that the algorithm's currently configuration is valid
bool
refine_detections_with_svm
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}


std::map<std::string, double> refine_detections_with_svm::apply_svms_from_directory(
    std::string dir_path, svm_node *x) const {

  LOG_ASSERT(logger(), boost::filesystem::is_directory(dir_path) == true,
             "Invalid path");

  boost::filesystem::path p(dir_path);
  boost::filesystem::directory_iterator it{p};

  boost::regex file_regex("^.*\\.svm");
  boost::cmatch char_matches;

  int *labels = new int[2];
  std::map <std::string, double> result;

  for (; it != boost::filesystem::directory_iterator {}; ++it) {
    std::string file_name_with_path = it -> path().string().c_str();
    std::string file_name = (it -> path()).filename().string();

    LOG_ASSERT(logger(), boost::regex_match(file_name.c_str(), char_matches, file_regex),
               "All svm files must have .svm extension");

    svm_model *model = svm_load_model(file_name_with_path.c_str());
    int num_classes = svm_get_nr_class(model);
    LOG_ASSERT(logger(), svm_check_probability_model(model), "Invalid Model");

    // We're expecting a two class problem
    LOG_ASSERT(logger(), num_classes == 2, "Invalid Model");
    svm_get_labels(model, labels);
    LOG_ASSERT(logger(), (labels[0] == 1 && labels[1] == -1) ||
        (labels[0] == -1 && labels[1] == 1), "Invalid Model");

    double *prob_estimates = new double[2];
    svm_predict_probability(model, x, prob_estimates);

    result[file_name] = labels[0] == 1 ? prob_estimates[0] : prob_estimates[1];
    delete[] prob_estimates;
    svm_free_and_destroy_model(&model);
  }
  delete[] labels;
  return result;
}


vital::detected_object_set_sptr
refine_detections_with_svm
::refine( vital::image_container_sptr image_data,
          vital::detected_object_set_sptr detections ) const
{

  if( !detections )
  {
    return detections;
  }

  for( auto det : *detections )
  {
    std::vector<double> descriptor_vector = det -> descriptor()-> as_double();
    size_t descriptor_size = descriptor_vector.size();
    svm_node *svm_nodes = new svm_node[descriptor_size];

    for( size_t i = 0; i < descriptor_size; ++i ) {
      svm_nodes[i].index = i + 1;
      svm_nodes[i].value = descriptor_vector.at(i);
    }
    std::string md = d_->model_dir;
    typedef  std::map<std::string, double> Map;

    Map res = apply_svms_from_directory(md, svm_nodes);

    // Set detected object type using map
    for(Map::iterator it = res.begin(); it != res.end(); ++it) {
      det -> type() -> set_score(it -> first, it -> second);
    }
    delete[] svm_nodes;
  }

  return detections;
}

} // end namespace svm
} // end namespace arrows
} // end namespace kwiver
