# Copyright (c) SenseTime. All Rights Reserved.
# Adapted for VIAME integration.

# Names of the two files the trainer wrapper and the training entry point pass
# validation between them by, both written in the train directory. They live
# here because the wrapper cannot import the entry point to reach them:
# importing that module parses the command line and would exit the host
# process on the first unrecognised argument.
#
# validation_sequences.txt  written by the wrapper, one held out sequence
#                           directory name per line
# validation_losses.txt     appended by the entry point, one line per epoch,
#                           read back by the wrapper to pick which checkpoint
#                           to ship
VALIDATION_SEQUENCES = 'validation_sequences.txt'
VALIDATION_RECORD = 'validation_losses.txt'
