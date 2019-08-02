#!/usr/bin/env bash

## Parses text file(s) and converts them to a list of -D flags to hand to Cmake
## Cmake will use the last version of the flag provided, so later flags
## overwrite earlier ones
## Pass in a sequence of paths of config files


errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

file_to_flags () {
    # Read a file, and convert argument lines into flags
    grep -ho '^[^#]*' $@ | sed 's/^/-D/' | tr '\n' ' '
}

for arg in "$@" ; do
    if [[ ! -e "${arg}" ]]; then
        errcho "${arg} not found. VIAME_VARIANT must have corresponding config"
    fi
done

flags="$(file_to_flags $@) "
echo $flags