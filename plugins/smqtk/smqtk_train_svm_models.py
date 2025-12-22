# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

 from __future__ import print_function

import smqtk.algorithms
import smqtk.iqr
import smqtk.representation
import smqtk.representation.descriptor_element.local_elements
import smqtk.utils.plugin

import json
import math
import svmutil
import svm
import os
import ctypes
import sys
import random
import numpy

default_desc_config = "/configs/pipelines/smqtk_desc_index.json"
default_nn_config = "/configs/pipelines/smqtk_nn_index.json"

default_desc_index = os.environ['VIAME_INSTALL'] + default_desc_config
default_nn_index = os.environ['VIAME_INSTALL'] + default_nn_config

def generate_svm_model( positive_uid_files, negative_uid_files,
  output_file, smqtk_params = dict() ):

  # Add default values for params if not present in input dict
  if not 'descriptor_index_config_file' in smqtk_params:
    smqtk_params['descriptor_index_config_file'] = default_desc_index
  if not 'neighbor_index_config_file' in smqtk_params:
    smqtk_params['neighbor_index_config_file'] = default_nn_index
  if not 'maximum_positive_count' in smqtk_params:
    smqtk_params['maximum_positive_count'] = 75
  if not 'maximum_negative_count' in smqtk_params:
    smqtk_params['maximum_negative_count'] = 750
  if not 'train_on_neighbors_only' in smqtk_params:
    smqtk_params['train_on_neighbors_only'] = False

  # Load indices
  print( " - Loading descriptor indices" )

  di_json_config_path = smqtk_params['descriptor_index_config_file']
  with open( di_json_config_path ) as f:
    di_json_config = json.load( f )

  # Path to the json config file for the NearestNeighborsIndex
  nn_json_config_path = smqtk_params['neighbor_index_config_file']
  with open( nn_json_config_path ) as f:
    nn_json_config = json.load( f )

  # Load positive samples for this class
  if len( positive_uid_files ) == 0:
    return

  pos_uuids = []
  neg_uuids = []

  for pos_file in positive_uid_files:
    for line in open( pos_file, "r" ):
      pos_uuids.append( line.rstrip() )

  pos_uuids = set( pos_uuids )

  for neg_file in negative_uid_files:
    for line in open( neg_file, "r" ):
      uuid = line.rstrip()
      if uuid not in pos_uuids:
        neg_uuids.append( uuid )

  if len( pos_uuids ) == 0 or len( neg_uuids ) == 0:
    print( "Error: Not enough training samples" )
    return

  # Set of descriptors to pull positive/negative querys from.
  descriptor_set = smqtk.utils.plugin.from_plugin_config(
    di_json_config,
    smqtk.representation.get_descriptor_index_impls()
  )

  # Nearest Neighbors index to use for IQR working index population.
  neighbor_index = smqtk.utils.plugin.from_plugin_config(
    nn_json_config,
    smqtk.algorithms.get_nn_index_impls()
  )

  # Max count threshold
  max_pos_samples = int( smqtk_params['maximum_positive_count'] )
  max_neg_samples = int( smqtk_params['maximum_negative_count'] )

  if len( pos_uuids ) > max_pos_samples:
    pos_uuids = random.sample( pos_uuids, max_pos_samples )

  pos_seed_neighbors = ( max_neg_samples / max_pos_samples )

  if smqtk_params['train_on_neighbors_only']:
    pos_seed_neighbors = pos_seed_neighbors * 2
  else:
    max_neg_samples = int( max_neg_samples / 2 )

  if pos_seed_neighbors < 2:
    pos_seed_neighbors = 2

  # Reset index on new query, a new query is one without IQR feedback
  iqr_session = smqtk.iqr.IqrSession( pos_seed_neighbors )
  pos_descrs = descriptor_set.get_many_descriptors( pos_uuids )

  print( " - Formatting data for initial search" )
  iqr_session.adjudicate( set( pos_descrs ) )

  iqr_session.update_working_index( neighbor_index )
  iqr_session.refine()

  # Perform 2nd round training on negatives from NN search
  print( " - Formatting data for SVM model train" )
  ordered_results = iqr_session.ordered_results()
  if ordered_results is None:
    ordered_results = []
  ordered_feedback = iqr_session.ordered_feedback()
  if ordered_feedback is None:
    ordered_feedback = []
  top_elems, top_dists = zip( *ordered_results )
  top_uuids = [ e.uuid() for e in top_elems ]
  feedback_uuids = [ e[0].uuid() for e in ordered_feedback ]

  best_neg_uuids = set( top_uuids ).intersection( neg_uuids )
  best_neg_uuids.update( set( feedback_uuids ).intersection( neg_uuids ) )

  if len( best_neg_uuids ) > max_neg_samples:
    best_neg_uuids = set( random.sample( best_neg_uuids, max_neg_samples ) )

  if not smqtk_params['train_on_neighbors_only']:
    if len( neg_uuids ) > max_neg_samples:
      best_neg_uuids.update( set( random.sample( neg_uuids, max_neg_samples ) ) )
    else:
      best_neg_uuids.update( set( neg_uuids ) )

  best_neg_descrs = descriptor_set.get_many_descriptors( best_neg_uuids )

  iqr_session.adjudicate( set( pos_descrs ), set( best_neg_descrs ) )

  print( " - Training SVM model" )
  print( "     + Positive sample count: " + str( len( pos_uuids ) ) )
  print( "     + Negative sample count: " + str( len( best_neg_uuids ) ) )
  iqr_session.update_working_index( neighbor_index )
  iqr_session.refine()

  try:
    svm_model = iqr_session.rel_index.get_model()
    svmutil.svm_save_model( output_file.encode(), svm_model )
  except:
    return

def generate_svm_models( folder = "database", id_extension = "lbl",
  background_id = "background", output_folder = "category_models",
  smqtk_params = dict() ):

  # Find all label files in input folder except background
  if not os.path.exists( folder ) and os.path.exists( folder + ".lnk" ):
    folder = folder + ".lnk"
  folder = folder if not os.path.islink( folder ) else os.readlink( folder )
  if not os.path.isdir( folder ):
    print( "Input folder \"" + folder + "\" does not exist" )
    return
  label_files = [
    os.path.join( folder, f ) for f in sorted( os.listdir( folder ) )
    if not f.startswith('.') and f.endswith( '.' + id_extension )
  ]

  # Error checking on inputs
  if len( label_files ) == 0:
    print( "No label files present in input folder \"" + folder + "\"" )
    return

  # Generate output folder
  if not os.path.exists( output_folder ):
    os.makedirs( output_folder )

  # Generate SVM model for each category
  all_categories = [
    os.path.splitext( os.path.splitext( os.path.basename( f ) )[0] )[0]
    for f in label_files
  ]

  for category in all_categories:
    if category == background_id:
      continue
    print( "Training model for category: " + category )
    positive_files = []
    negative_files = []
    output_file = output_folder + '/' + category + '.svm'
    for label_file in label_files:
      if category + '.' in label_file:
        positive_files.append( label_file )
      else:
        negative_files.append( label_file )
    print( " - Positive files " + str( positive_files ) )
    generate_svm_model( positive_files, negative_files, output_file, smqtk_params )
