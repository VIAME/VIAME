#!/bin/bash -x
#
# Test script for merge_detection_sets_process
#
rm -f junk.xxx

kwiver runner ./test_merge_detection_sets.pipe

if ! diff -I '#.*'  junk.xxx test_merge_detection_sets.truth >/dev/null 2>&1
then
    echo "test_merge_process output file does not match expected"
fi

rm -f junk.xxx
