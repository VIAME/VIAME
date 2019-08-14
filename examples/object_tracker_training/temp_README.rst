==================================
SiamRPN++ Object Tracking Training
==================================

Here are the noticeable differences between my script, `train_siamrpn++_viame_csv.sh`, and the similar object detector training script:
  * `DATA_FOLDER` is the name of parent folder holding all the data folders, similar to `training_data_mouss/` in `../object_detector_training/`.
  * `MODEL_FOLDER` is the name of new folder that will appear when training starts, holding the following files/folders:
    - `crop511/`: a folder holding new versions of the input images, cropped as the model needs it.
    - `logs/`: a folder holding all of the training output in `logs.txt`.
    - `snapshot/`: a folder that will hold all of the model training checkpoints.
    - `dataset.json`: a json file that has all of the dataset's information formatted as the model needs it.
  * `NUM_PROC` is the number of (parallel) training processes; it splits evenly on multiple GPUs if more than 1 is available.
  * `--skip-crop` lets you skip the dataset cropping step if you've already done it.
