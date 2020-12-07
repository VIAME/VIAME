#!/bin/sh
# TODO write a .bat file
# Input locations and types
export INPUT_DIRECTORY="training_data_adaboost"
export TEMP_DIR="temp_dir"
export FEATURE_CONFIG="~/dev/VIAME/build/install/configs/pipelines/burnout_train_classifier.conf" 

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname $0)" && pwd)/../.."

echo ${VIAME_INSTALL}
source ${VIAME_INSTALL}/setup_viame.sh

remove_metadata_burnin -c ${FEATURE_CONFIG}

## Run pipeline
#viame_train_detector \
#  -i ${INPUT_DIRECTORY} \
#  -c ${VIAME_INSTALL}/configs/pipelines/train_adaboost_pixel_classifier.viame_csv.conf \
#  --threshold 0.0 # I don't know if this is needed
#feature_commands = ["{} -c {} \\".format(self._burnin_exec, self._feature_pipeline),
#                    "    --extract-features {} \\".format(
#                        self._temp_dir),
#                    "    --gt-image-type png \\",
#                    "    --feature-file {}/features.txt".format(
#                        self._temp_dir)]
#
#feature_command = "\n".join(feature_commands)
#model_file = os.path.join(
#    self._training_writer.write_path(), "trained_classifier.adb")
#
#
#train_command = "train_pixel_model {}/features.txt {} --positive-identifiers {} --negative-identifiers {} --max-iter-count {}".format(
#    self._temp_dir, model_file, self._positive_identifiers,
#    self._negative_identifers, self._max_iter_count)
#
## TODO migrate to subprocess.run()
#print(feature_command)
#os.system(feature_command)
#print(train_command)
#os.system(train_command)
#final_model_folder = os.path.join(os.getcwd(), "category_models")
#os.makedirs(final_model_folder, exist_ok=True)
#final_model_file = os.path.join(
#    final_model_folder, "trained_classifier.adb")
#try:
#    shutil.copyfile(model_file, final_model_file)
#except FileNotFoundError:
#    print("Final model was not produced to {}".format(model_file))