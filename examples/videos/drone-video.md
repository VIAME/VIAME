Instructions for stabilizing video from a drone.

After you get the video file off the camera on the drone, convert AVI
file to individual images for each frame using the following command.

ffmpeg -i MOVI0003.avi -f image2 MOVI0003-frames/MOVI0000-%05d.png

This command converts or, using the technical term, decodes the input
video (e.g. MOVI0003.avi) into png images. The output images are
sequentially numbered so they can be easily processed in time order.

Now that you have all the individual images available, look through
them and remove the images that are not suitable for processing, such
the close up images of the lawn, or the wild crash at the end.

Make a file that contains the file names of images you want to
process, one per line. The easiest way to do this is to list the files
to a directory (e.g. ls *.png > image_list.txt). This list will be used
to drive the image processing.

The next step is to edit the configuration file to select the list of
frames and specify the output parameters. We will be using the
images_to_kwa.pipe configuration file as an example.

The list of images to process is specified in the configuration
section for the input process. Put the name of the image list file in
the entry for ":image_list_file" as either a relative or absolute
path.

The generated outputs are specified in writer process at the bottom of
the configuration file using the ":output_directory" and
":base_filename" entries.  A kw-archive output consists of several
files that will be written to the specified output_directory with the
supplied base file name. The files will be differentiated by their
extension.

There are plenty of other configuration values in that file, but these
are the only ones you need to worry about to get started.

Now you are ready to generate the kw_archive output by running the
following command with the configuration/pipeline file.

kwiver runner ../source/kwiver/pipeline_configs/images_to_kwa.pipe

The resulting output can be displayed with the vsPlay tool.
