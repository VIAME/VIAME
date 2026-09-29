# Command Line Interface

Everything VIAME does can be run from a terminal through the `viame` command. It suits batch processing, machines without a display, and scripted workflows. The launch scripts in the example and project folders call the same command.

## Setup

In a desktop installation, load the VIAME environment into the terminal first:

```bash
# Linux and Mac
source /path/to/viame/setup_viame.sh
```

```
REM Windows
call C:\path\to\viame\setup_viame.bat
```

When VIAME was installed with `pip install viame`, the command is available without this step.

## Usage

```bash
viame <command> [arguments]
```

`viame help` lists every command, and `viame <command> --help` shows the options of one.

## Commands

### Running and evaluating models

| Command | Purpose |
|----|----|
| `run` | Process videos or images, or run a single pipeline file |
| `runner` | Run a pipeline file |
| `train` | Train detector or tracker models |
| `monitor` | Monitor a training run and report progress by log or email |
| `ensemble` | Learn fusion parameters for ensembling multiple detectors |
| `score` | Score detection and tracking results against groundtruth |
| `plot` | Plot detection counts per frame, or evaluation results |
| `segment` | Add SAM2 segmentation polygons to an existing box-level annotation set |

### Working with files

| Command | Purpose |
|----|----|
| `convert` | Convert annotation, calibration and registration files between formats |
| `csv` | Perform filtering and analysis actions on VIAME CSV files |
| `json` | Perform filtering and analysis actions on DIVE and COCO JSON files |
| `extract` | Extract frames from video files |
| `resample` | Resample object tracks from one frame rate to another |
| `inspect` | Identify a file, check it is intact, and say how VIAME can use it |
| `metadata` | Dump unified per-image survey metadata for a site folder |

### Search

| Command | Purpose |
|----|----|
| `index` | Build and manage the video search index: add, remove, build, list, status, hash |
| `database` | Initialize, start, stop and index the descriptor database |
| `search` | Launch the video search (query) interface |

### Stereo, registration and 3D

| Command | Purpose |
|----|----|
| `calibrate` | Estimate stereo calibration from calibration target images |
| `rectify` | Rectify a stereo image pair using calibration parameters |
| `disparity` | Estimate disparity between a pair of rectified images |
| `depth` | Estimate depth from a pair of rectified images |
| `register` | Register survey imagery and detect previously-observed regions |
| `mosaic` | Stitch a mosaic from images and their homographies |
| `3d` | Build a 3D model from UAS imagery |

### Pipelines and configuration

| Command | Purpose |
|----|----|
| `pipeline` | Generate, inspect, validate and modify pipeline files |
| `pipe-config` | Configure a pipeline |
| `pipe-to-dot` | Write the layout of a pipeline as a DOT graph |
| `pipe-gui` | Run pipelines in a simple interface |
| `configs` | Extract pipeline and training parameters as JSON |
| `explore-config` | Explore the configuration of an algorithm |

### System

| Command | Purpose |
|----|----|
| `gpu` | Check GPU properties of the system |
| `add-ons` | List installed add-on model packs and download new ones |

## Common Tasks

### Running a pipeline

`viame run` takes a pipeline and the data to process. The pipeline can be a file, or the name of one in `configs/pipelines`. The data can be a video, an image, a text file listing images, or a folder.

```bash
viame run detector_generic_proposals my_video.mp4
```

A folder of videos or image folders is processed in one call:

```bash
viame run -d my_data_folder -p detector_generic_proposals.pipe
```

A model file can be given in place of a pipeline. It is wrapped in the default detector pipeline, or the frame classifier pipeline for a classifier:

```bash
viame run trained_model.zip my_video.mp4
```

### Changing a setting for one run

Any pipeline setting can be overridden with `-s`, named by its process and setting, without editing the pipeline file:

```bash
viame run detector_generic_proposals.pipe -s input:video_filename=my_images.txt
```

### Training a model

```bash
viame train -i training_data -c train_detector_default.conf
```

`viame train --list` shows every trainable algorithm. See [detector training](https://viame.github.io/VIAME/sections/object_detector_training.html) for the layout of the training data and the available configurations.

### Scoring results

```bash
viame score computed_detections.csv groundtruth.csv --per-class
```

See [scoring detectors and trackers](https://viame.github.io/VIAME/sections/scoring_and_evaluation.html).

### Converting annotations

The output format is taken from the file extension:

```bash
viame convert annotations.csv annotations.json
```

See [detection file conversions](https://viame.github.io/VIAME/sections/detection_file_conversions.html).
