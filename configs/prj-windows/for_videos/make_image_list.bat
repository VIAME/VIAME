
REM Processing options
SET INPUT_DIRECTORY=frames

REM Make list from glob pattern
dir /s/b/o "%INPUT_DIRECTORY%\*" > input_list.txt
