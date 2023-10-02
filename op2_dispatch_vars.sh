#!/usr/bin/env bash

######################################
# DO NOT CHANGE THIS FOLLOWING LINE: #
OP2_BASELINE_FILE="baseline_op_02.c" #
######################################

############################################
# HOWEVER, CHANGE THESE LINES:             #
# Replace the filenames with your variants #
############################################
OP2_SUBMISSION_VAR01_FILE="tuned_variant01_op_02.c" # <-- CHANGE ME!
OP2_SUBMISSION_VAR02_FILE="tuned_variant02_op_02.c" # <-- CHANGE ME!
OP2_SUBMISSION_VAR03_FILE="tuned_variant03_op_02.c" # <-- CHANGE ME!
OP2_SUBMISSION_VAR04_FILE="tuned_variant04_op_02.c" # <-- CHANGE ME!
OP2_SUBMISSION_VAR05_FILE="tuned_variant05_op_02.c" # <-- CHANGE ME!
OP2_SUBMISSION_VAR06_FILE="tuned_variant06_op_02.c" # <-- CHANGE ME!

######################################################
# You can even change the compiler flags if you want #
######################################################
# CFLAGS="-std=c99 -O2"
CFLAGS="-std=c99 -O2 -mavx2 -fopenmp -mfma"

