
export VIAME_INSTALL=../..
export INPUT_IMAGE_PATTERN=*.png

ls ${INPUT_IMAGE_GLOB} > input_list.txt

kwiver runner 

python create_mosaic.py 
