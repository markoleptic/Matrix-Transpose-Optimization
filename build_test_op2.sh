#!/usr/bin/env bash
#
# This file builds the verifier code based on the configured vars.
# You should not need to modify this.
#
# - richard.m.veras@ou.edu

source op2_dispatch_vars.sh

echo $OP2_BASELINE_FILE

echo $OP2_SUBMISSION_VAR01_FILE
echo $OP2_SUBMISSION_VAR02_FILE
echo $OP2_SUBMISSION_VAR03_FILE

echo $CFLAGS

# Build each pair of baseline with variant
# gcc -std=c99 verify_op_02.c baseline_op_02.c -o run.x

# Build the verifier code
gcc -std=c99 -c -DFUN_NAME_REF="baseline" -DFUN_NAME_TST="test" verify_op_02.c 

# Build the reference baseline
gcc -std=c99 -c -DFUN_NAME="baseline" baseline_op_02.c

# Build the variants
gcc $CFLAGS -c -DFUN_NAME="test" $OP2_SUBMISSION_VAR01_FILE -o op2_var01.o
gcc $CFLAGS -c -DFUN_NAME="test" $OP2_SUBMISSION_VAR02_FILE -o op2_var02.o
gcc $CFLAGS -c -DFUN_NAME="test" $OP2_SUBMISSION_VAR02_FILE -o op2_var03.o

# Build the verifier
gcc -std=c99 verify_op_02.o baseline_op_02.o op2_var01.o -o ./run_test_op2_var01.x
gcc -std=c99 verify_op_02.o baseline_op_02.o op2_var02.o -o ./run_test_op2_var02.x
gcc -std=c99 verify_op_02.o baseline_op_02.o op2_var03.o -o ./run_test_op2_var03.x
