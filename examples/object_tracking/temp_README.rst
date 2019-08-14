
========================
PYSOT Object Tracking
========================

The main file of note is `../../configs/pipelines/pysot_common_default_tracker.pipe`
1. `:seed_track` is the starting bounding box of the object; note that (as of now) the script assumes the object is on frame 1.  In hindsight, that should be another parameter.
2. `:track_th` is the confidence threshold for stopping a track, only needed if using vpView.
3. `:ots_input_flag` is a flag that tells KWIVER that an `object_track_state` is input, only needed for vpView.
4. Make sure to choose the model (config and weights) that you want to use; these should be in `configs/pipelines/models/` after building.
