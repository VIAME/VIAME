
=============================
CLI Tools and Example Folders
=============================

In the '[install]/examples' folder, there are a number of subfolders, with each folder corresponding
to a different core functionality. The scripts in each of these folders can be copied to and run
from any directory on your computer, the only item requiring change being the 'VIAME_INSTALL' path at
the top of the each run script. These scripts can be opened and edited in any text editor to point
the VIAME_INSTALL path to the location of your installed (or built) binaries. This is true on both
Windows, Linux, and Mac.

The 'examples' folder is one of two core entry points into running VIAME functionality. The other is
to copy project files for your operating system, '[install]/configs/prj-linux' or
'[install]/configs/prj-windows' to a directory of your choice and run things from there. Not all
functionality is in the default project file scripts, however, but it is a good entry point if you
just want to get started on object detection and/or tracking.

Each example is run in a different fashion, but there are 3 core commands you need to know in
order to run them on Linux:

'bash' - for running commands, e.g. 'bash run_annotation_gui.sh' which launches the application

'ls' - for making file lists of images to process, e.g. 'ls *.png > input_list.txt' to list all
png image files in a folder

'cd' - go into an example directory, e.g. 'cd annotation_and_visualization' to move down into the
annotation_and_visualization example directory. 'cd ..' is another useful command which moves one
directory up, alongside a lone 'ls' command to list all files in the current directory.

To run the examples on Windows, you just need to be able to run (double click) the .bat scripts
in the given directories. Additionally, knowing how to make a list of files, e.g. 'dir > filename.txt'
on the windows command line can also be useful for processing custom image lists.

