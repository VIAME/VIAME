#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os
import logging

from flask import Flask, jsonify, request

from train_handler import TrainHandler
from process_handler import ProcessHandler

app = Flask( __name__ )

logging.basicConfig( level=logging.INFO )
logger = logging.getLogger( "viame.vertex-ai" )

# Vertex AI environment variables
AIP_HTTP_PORT = int( os.environ.get( "AIP_HTTP_PORT", "8080" ) )
AIP_PREDICT_ROUTE = os.environ.get( "AIP_PREDICT_ROUTE", "/predict" )
AIP_HEALTH_ROUTE = os.environ.get( "AIP_HEALTH_ROUTE", "/health" )
AIP_STORAGE_URI = os.environ.get( "AIP_STORAGE_URI", "" )
AIP_MODE = os.environ.get( "AIP_MODE", "" )

# VIAME-specific configuration
VIAME_INSTALL_DIR = os.environ.get( "VIAME_INSTALL_DIR", "/opt/noaa/viame" )
WORK_DIR = os.environ.get( "VIAME_WORK_DIR", "/workspace" )
PIPELINE = os.environ.get( "VIAME_PIPELINE", "" )
TRAIN_CONFIG = os.environ.get( "VIAME_TRAIN_CONFIG", "" )

# Initialize handlers
train_handler = TrainHandler(
  viame_dir=VIAME_INSTALL_DIR,
  work_dir=WORK_DIR,
  model_storage_uri=AIP_STORAGE_URI
)

process_handler = ProcessHandler(
  viame_dir=VIAME_INSTALL_DIR,
  work_dir=WORK_DIR,
  default_pipeline=PIPELINE,
  model_storage_uri=AIP_STORAGE_URI
)


@app.route( AIP_HEALTH_ROUTE, methods=[ "GET" ] )
def health():
  """Vertex AI health check - return 200 when ready to serve."""
  return jsonify( { "status": "healthy" } ), 200


@app.route( AIP_PREDICT_ROUTE, methods=[ "POST" ] )
def predict():
  """Handle prediction requests from Vertex AI.

  Expected request body:
  {
    "instances": [
      {
        "input_path": "gs://bucket/video.mp4",
        "pipeline": "detector_yolo.pipe",
        "settings": { "key": "value" }
      }
    ],
    "parameters": {
      "frame_rate": "5",
      "output_format": "viame_csv"
    }
  }
  """
  payload = request.get_json( silent=True )

  if not payload or "instances" not in payload:
    return jsonify( { "error": "Missing 'instances' in request body" } ), 400

  instances = payload[ "instances" ]
  parameters = payload.get( "parameters", {} )

  try:
    predictions = process_handler.run( instances, parameters )
    return jsonify( { "predictions": predictions } )
  except Exception as e:
    logger.error( "Prediction failed: %s", str( e ) )
    return jsonify( { "error": str( e ) } ), 500


@app.route( "/train", methods=[ "POST" ] )
def train():
  """Handle training requests.

  Expected request body:
  {
    "input_dir": "gs://bucket/training-data/",
    "output_dir": "gs://bucket/models/",
    "config": "train_yolo.conf",
    "settings": { "key": "value" }
  }
  """
  payload = request.get_json( silent=True )

  if not payload:
    return jsonify( { "error": "Missing request body" } ), 400

  try:
    result = train_handler.run( payload )
    return jsonify( result )
  except Exception as e:
    logger.error( "Training failed: %s", str( e ) )
    return jsonify( { "error": str( e ) } ), 500


@app.route( "/status", methods=[ "GET" ] )
def status():
  """Check status of a running train or process job."""
  return jsonify( {
    "train": train_handler.get_status(),
    "process": process_handler.get_status()
  } )


if __name__ == "__main__":
  if AIP_STORAGE_URI:
    logger.info( "Downloading model artifacts from %s", AIP_STORAGE_URI )
    process_handler.download_model_artifacts( AIP_STORAGE_URI )

  logger.info( "Starting VIAME Vertex AI server on port %d", AIP_HTTP_PORT )
  app.run( host="0.0.0.0", port=AIP_HTTP_PORT )
