# Scripts and Example Folders

In the [examples](https://github.com/VIAME/VIAME/tree/main/examples) folder of a desktop installation there are a number of subfolders, with each folder corresponding to a different major functionality of VIAME. The scripts in each of these folders can be run as-is in each folder, or alternatively copied, edited, and run from any directory on your computer. Generally speaking, for just getting familiar with the tools it is okay to run them from the installation folders, but for performing any real work it is best to copy them outside of the installers to another location, as some operations (e.g. model training) generate a lot of additional temporary files and you might forget they are there if calling them from within the installers.

Each script calls a command line interface (CLI) executable to perform some function. The alternative to running CLI tools, is to run algorithms through graphical user interfaces within VIAME, such as DIVE (see [User Interfaces](https://viame.github.io/VIAME/sections/user_interfaces.html)). User interfaces support a large number of algorithms, though not everything found within scripts and examples. Lastly, [Project Folders](https://viame.github.io/VIAME/sections/project_folders.html) provide multiple scripts in one location for different stages of an object-detector-training lifecycle for users who prefer using them.

To run the examples on Windows, you need to be able to run (double click) the .bat scripts in the given directories (see image below). Additionally, knowing how to make a list of files, e.g. "dir \> filename.txt" on the windows command line can possibly be useful for processing custom image lists.

To run the examples on Linux, there are 3 core commands you need to know:

"bash" - for running commands, e.g. "bash run_annotation_gui.sh" which launches the application

"ls" - for making file lists of images to process, e.g. "ls \*.png \> input_list.txt" to list all png image files in a folder

"cd" - go into an example directory, e.g. "cd user_interfaces" to move down into the user_interfaces example directory. "cd .." is another useful command which moves one directory up, alongside a lone "ls" command to list all files in the current directory.

In all of the documentation for each example, the ".bat" or ".sh" extension is omitted, as the extension is operating system (OS) dependent. The functionality and scripts are nearly identical across different OSes, however.

## Editing Scripts

All scripts can be opened and edited in any text editor to adjust script options. On window, the script can be right-clicked and then "Edit" can be selected to open it in the default windows text editor (Notepad). Using a different editor like Notepad++ is often recommended. On Linux, editors such as Emacs, Vim, or GEdit can be useful for this.

For example, one of the most common values that requires editing is the "VIAME_INSTALL" path at the top of each run script. This is useful if copying and running any scripts outside of project folders or the example directories. The VIAME_INSTALL path should point to the location of your installed (or compiled) binaries. In the case of installers, this is the top level "VIAME" folder. In the case of manual software builds, this would be the "install" directory within a build tree.

Other common options are "INPUT_FOLDER", a directory containing multiple input videos or images, and "DEFAULT_FRAME_RATE", the default frames per second to run algorithms on if the input is a video instead of images. The desired input options should be set prior to running the scripts.

## Bulk Processing Scripts

Each .sh or .bat script in the example folder is designed to run on either a single sequence of data (e.g. one video, one image sequence, or one image list) or alternatively a folder or folder of folders containing many sequences. Inside each script is either a call to "viame run" (the most common default), "viame", or the "kwiver" executable. The first two are meant for bulk processing multiple sequences, while the latter only processes a single sequence.

Depending on the scripts, inputs may just be raw data (images or video), raw data plus annotation files (.csv, .json), or raw data plus metadata and annotations. In the default case, you can have any number of videos in the input folder, or image folders containing multiple images. For example:

```
input_folder/         <-- root folder
    video1.mpg
    video2.mpg
    sequence3/        <-- subfolder
        image1.png
        image2.png
        image3.png
        etc...
```

Images and videos can optionally be mixed in the same input folder.

Alternatively, the input folder can just be a folder of images:

```
input_folder/     <-- root folder
    image1.png
    image2.png
    image3.png
    etc...
```

Or just videos:

```
input_folder/     <-- root folder
    video1.mpg
    video2.mpg
```

Annotation files can provide either metadata (e.g. the FPS to process videos at, different on a per-video basis) or boxes/categories for training different types of AI models. For cases that require annotation files alongside videos, they should be in the same directory as the video with the same name, except instead of the video extension it should be a .csv or .json file. For image sequences, there should be a single annotation file of any name in the folder of images. Alternatively, an annotation file as the same name as the input image sequence folder can be placed at the same directory level of the folder.

For example:

```
input_folder/           <-- root folder
    video1.mpg
    video1.csv
    video2.mpg
    video2.csv
    sequence3/          <-- subfolder
        image1.png
        image2.png
        image3.png
        whatever.csv
```

is a valid input

```
input_folder/         <-- root folder
    video1.mpg
    video1.json
    video2.mpg
    video2.json
    sequence3.json
    sequence3/        <-- subfolder
        image1.png
        image2.png
        image3.png
```

is also a valid input

```
input_folder/          <-- root folder
    video1.mpg
    video1.json
    video2.mpg
    video2.json
    sequence3/         <-- subfolder
        image1.png
        image2.png
        image3.png
        truth1.json
        truth2.json
```

is not a valid input, as the image folder contains two possible truth files, and that will confuse the input loader. An input folder without a truth file will also error out with a hard error.

## Scripts vs Direct Executable Calls

All of the scripts within examples or project folders call the following executables under the hood. These can be called by more advanced users. Running "-?" or "-help" on each script shows a list of all potential options.

viame run - bulk runs a particular algorithmic pipeline on multiple files

kwiver - runs a single pipeline on multiple files

any of the python scripts in the configs directory - contain specialized functionality such as running camera calibration, generating mosaics, or running algorithm evaluation code, all in standalone scripts

viame - Command line tool for running pipelines and training models

<!-- project-folders -->

## Project Folders

The "examples" folder is one of two core entry points into running VIAME functionality. The other is a project folder: a folder of scripts which is copied to a working drive outside of the installation, kept beside your own data, and run from there. A template project folder is located in the "configs/prj-template" folder of a desktop installation, holding .bat scripts on Windows and .sh scripts on Linux.

### Running VIAME from Scripts

Each script is a short text file which sets a few options and then makes a single call to the `viame` command line tool, so running a script does the same thing as typing that command out by hand. To run one:

1.  Copy "configs/prj-template" to a folder of your own, on a drive with room for your data and the outputs
2.  Open the script in a text editor and check the options at its top, which are listed below. `VIAME_INSTALL` has to point to your installation, and is set to "/opt/noaa/viame" on Linux and "C:\Program Files\VIAME" on Windows by default
3.  Put your data where the script looks for it, inside of the project folder. This is a "videos" folder for the processing scripts and a "training_data" folder for the training ones. See [scripts and example folders](https://viame.github.io/VIAME/sections/examples_overview.html) for how these folders are laid out
4.  Run the script from within the project folder, as the folders named in it are relative to where it is run. On Windows, double click the .bat file. On Linux, open a terminal in the project folder and run, for example, `bash generate_detections_using_default_model.sh`

Results are written into the project folder, in the "output" folder unless the script says otherwise.

The options at the top of the scripts are:

| Option | Description |
|----|----|
| `VIAME_INSTALL` | Location of the VIAME installation |
| `INPUT` | Folder of videos, or of folders of images, to process ("videos" by default) |
| `OUTPUT` | Folder the results are written into ("output" by default) |
| `INPUT_DIRECTORY` | Folder of annotated data to train on ("training_data" by default) |
| `FRAME_RATE` | Frames per second to process videos at |
| `PIPELINE` | Pipeline to run over each input, for the scripts which use a default model |
| `TOTAL_GPU_COUNT` | Number of GPUs to spread the processing over |
| `PIPES_PER_GPU` | Number of inputs to process at once on each GPU |

Not every script has every option, and anything else the `viame` tool accepts can be added to the command at the bottom of a script, see the [command line interface](https://viame.github.io/VIAME/sections/command_line_interface.html) page.

### Scripts in a Project Folder

| Task | Scripts |
|----|----|
| Annotate data | launch_dive_interface |
| Prepare video | extract_video_frames, extract_video_clips, make_image_list |
| Run a model shipped with VIAME | generate_detections_using_default_model, generate_tracks_using_default_model |
| Train a model on your annotations | train_deep_cfrnn_detector_from_csv, train_deep_yolo_detector_from_csv, train_svm_detector_from_csv, continue_training_deep_cfrnn_detector |
| Run a model you trained | generate_detections_using_trained_model, generate_tracks_using_trained_model, classify_frames_using_trained_model |
| Search an archive | create_index.around_detections, create_index.detection_and_tracking, launch_search_interface, process_database_using_svm_model |
| Build mosaics | generate_mosaics_using_default_model |

A few of the scripts differ between Windows and Linux. A common order to use them in is to annotate a few videos in [DIVE](https://viame.github.io/VIAME/sections/dive/index.html), export the annotations into "training_data" beside their videos, train a detector, and then run the trained model over the rest of the videos.

### Trained Models

Training writes the trained model as a single pack, "trained_model.zip", holding the generated pipelines, model files, a model card and any evaluation results. The "\*\_using_trained_model", "run_trained_model" and "continue_training" scripts use that pack when present and otherwise fall back to an unpacked "trained_model" folder (or "category_models" from older releases). Run one pipeline from a pack with "viame run -p trained_model.zip/detector.pipe". SVM training still writes its models to a "trained_model" folder, which the SVM pipelines read directly.

Not all functionality is in the default project file scripts, however, but it is a good entry point if you just want to get started on object detection and/or tracking.
