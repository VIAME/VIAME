======================
PYSOT Object Tracking
======================

The main file of note is `../../configs/pipelines/pysot_common_default_tracker.pipe`
  1. `:seed_track` is the starting bounding box of the object; note that (as of now) the script assumes the object is on frame 1.  In hindsight, that should be another parameter.
  2. `:track_th` is the confidence threshold for stopping a track, only needed if using vpView.
  3. `:ots_input_flag` is a flag that tells KWIVER that an `object_track_state` is input, only needed for vpView.
  4. Make sure to choose the model (config and weights) that you want to use; these should be in `configs/pipelines/models/` after building.

Additionally, you need to add the file https://kwgitlab.kitware.com/computer-vision/KWIVER/blob/vcat/dev/add-pysot/arrows/pytorch/pysot_tracker.py to `lib/python3.6/site-packages/kwiver/processes/pytorch/` since it's only available in my KWIVER branch.


When using the pysot tracker in the GUI (vpView), here are the basic instructions:
  1. Make a starting bounding box using the GUI
    * If there are already existing tracks, the starting track must be the largest number (bottom of the list.)
    * The starting frame can be any frame (not just the first) but note that it will still process (and do nothing with) all frames before it when the plugin runs.
  2. Run the pysot plugin (called `pysot_tracker`)
    * Note that the plugin will start from frame 1 and run until the last frame by default, even if it stops tracking before then.
    * If it's already gotten past all frames where the object is, hitting cancel stops the plugin but still makes a new track with all frames until that point. 
  3. You'll see a new track appear when it finishes -- it has the starting track included, so feel free to remove the initial single-frame track.