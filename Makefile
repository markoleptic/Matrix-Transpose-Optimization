# This Makefile orchestrates the building, verification and timing of
# your implementations.
#
# - richard.m.veras@ou.edu

# Increments to use in the tests
MIN=16
MAX=256
#MAX=64
STEP=16

all: run_verifier run_bench

run_verifier: run_verifier_op2_var01 run_verifier_op2_var02 run_verifier_op2_var03

run_bench: run_bench_op2_var01 run_bench_op2_var02

run_bench_op2_var01: build_bench
	touch result_bench_op2_var01.csv
        # square matrices, a and b are row major
        #                       min    max    step    m n   rs_src cs_src rs_dst cs_dst
	./run_bench_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     1      -1 | tee -a result_bench_op2_var01.csv
        # square matrices, a and b are col major
	./run_bench_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     -1      1 | tee -a result_bench_op2_var01.csv
        # square matrices, a row major and b col major
	./run_bench_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     -1      1 | tee -a result_bench_op2_var01.csv
        # square matrices, a col major and b row major
	./run_bench_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     1      -1 | tee -a result_bench_op2_var01.csv
        # square matrices, a and b general strides
	./run_bench_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 1   -2      1     1      -2 | tee -a result_bench_op2_var01.csv
        # rectangular matrices, a row major and b col major
	./run_bench_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 -${MIN}   1      -1     -1      1 | tee -a result_bench_op2_var01.csv
        # rectangular matrices, a row major and b col major
	./run_bench_op2_var01.x  ${MIN} ${MAX} ${STEP} -${MIN} 1   1      -1     -1      1 | tee -a result_bench_op2_var01.csv

run_bench_op2_var02: build_bench
	touch result_bench_op2_var02.csv
        # square matrices, a and b are row major
        #                       min    max    step    m n   rs_src cs_src rs_dst cs_dst
	./run_bench_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     1      -1 | tee -a result_bench_op2_var02.csv
        # square matrices, a and b are col major
	./run_bench_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     -1      1 | tee -a result_bench_op2_var02.csv
        # square matrices, a row major and b col major
	./run_bench_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     -1      1 | tee -a result_bench_op2_var02.csv
        # square matrices, a col major and b row major
	./run_bench_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     1      -1 | tee -a result_bench_op2_var02.csv
        # square matrices, a and b general strides
	./run_bench_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 1   -2      1     1      -2 | tee -a result_bench_op2_var02.csv
        # rectangular matrices, a row major and b col major
	./run_bench_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 -${MIN}   1      -1     -1      1 | tee -a result_bench_op2_var02.csv
        # rectangular matrices, a row major and b col major
	./run_bench_op2_var02.x  ${MIN} ${MAX} ${STEP} -${MIN} 1   1      -1     -1      1 | tee -a result_bench_op2_var02.csv

run_bench_op2_var03: build_bench
	touch result_bench_op2_var03.csv
        # square matrices, a and b are row major
        #                       min    max    step    m n   rs_src cs_src rs_dst cs_dst
	./run_bench_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     1      -1 | tee -a result_bench_op2_var03.csv
        # square matrices, a and b are col major
	./run_bench_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     -1      1 | tee -a result_bench_op2_var03.csv
        # square matrices, a row major and b col major
	./run_bench_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     -1      1 | tee -a result_bench_op2_var03.csv
        # square matrices, a col major and b row major
	./run_bench_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     1      -1 | tee -a result_bench_op2_var03.csv
        # square matrices, a and b general strides
	./run_bench_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 1   -2      1     1      -2 | tee -a result_bench_op2_var03.csv
        # rectangular matrices, a row major and b col major
	./run_bench_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 -${MIN}   1      -1     -1      1 | tee -a result_bench_op2_var03.csv
        # rectangular matrices, a row major and b col major
	./run_bench_op2_var03.x  ${MIN} ${MAX} ${STEP} -${MIN} 1   1      -1     -1      1 | tee -a result_bench_op2_var03.csv







run_verifier_op2_var01: build_verifier
	touch result_verification_op2_var01.csv
        # square matrices, a and b are row major
        #                       min    max    step    m n   rs_src cs_src rs_dst cs_dst
	./run_test_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     1      -1 | tee -a result_verification_op2_var01.csv
        # square matrices, a and b are col major
	./run_test_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     -1      1 | tee -a result_verification_op2_var01.csv
        # square matrices, a row major and b col major
	./run_test_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     -1      1 | tee -a result_verification_op2_var01.csv
        # square matrices, a col major and b row major
	./run_test_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     1      -1 | tee -a result_verification_op2_var01.csv
        # square matrices, a and b general strides
	./run_test_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 1   -2      1     1      -2 | tee -a result_verification_op2_var01.csv
        # rectangular matrices, a row major and b col major
	./run_test_op2_var01.x  ${MIN} ${MAX} ${STEP} 1 -${MIN}   1      -1     -1      1 | tee -a result_verification_op2_var01.csv
        # rectangular matrices, a row major and b col major
	./run_test_op2_var01.x  ${MIN} ${MAX} ${STEP} -${MIN} 1   1      -1     -1      1 | tee -a result_verification_op2_var01.csv
	grep -i "FAIL" result_verification_op2_var01.csv | wc -l

run_verifier_op2_var02: build_verifier
	touch result_verification_op2_var02.csv
        # square matrices, a and b are row major
        #                       min    max    step    m n   rs_src cs_src rs_dst cs_dst
	./run_test_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     1      -1 | tee -a result_verification_op2_var02.csv
        # square matrices, a and b are col major
	./run_test_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     -1      1 | tee -a result_verification_op2_var02.csv
        # square matrices, a row major and b col major
	./run_test_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     -1      1 | tee -a result_verification_op2_var02.csv
        # square matrices, a col major and b row major
	./run_test_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     1      -1 | tee -a result_verification_op2_var02.csv
        # square matrices, a and b general strides
	./run_test_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 1   -2      1     1      -2 | tee -a result_verification_op2_var02.csv
        # rectangular matrices, a row major and b col major
	./run_test_op2_var02.x  ${MIN} ${MAX} ${STEP} 1 -${MIN}   1      -1     -1      1 | tee -a result_verification_op2_var02.csv
        # rectangular matrices, a row major and b col major
	./run_test_op2_var02.x  ${MIN} ${MAX} ${STEP} -${MIN} 1   1      -1     -1      1 | tee -a result_verification_op2_var02.csv
	grep -i "FAIL" result_verification_op2_var02.csv | wc -l

run_verifier_op2_var03: build_verifier
	touch result_verification_op2_var03.csv
        # square matrices, a and b are row major
        #                       min    max    step    m n   rs_src cs_src rs_dst cs_dst
	./run_test_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     1      -1 | tee -a result_verification_op2_var03.csv
        # square matrices, a and b are col major
	./run_test_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     -1      1 | tee -a result_verification_op2_var03.csv
        # square matrices, a row major and b col major
	./run_test_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 1   1      -1     -1      1 | tee -a result_verification_op2_var03.csv
        # square matrices, a col major and b row major
	./run_test_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 1   -1      1     1      -1 | tee -a result_verification_op2_var03.csv
        # square matrices, a and b general strides
	./run_test_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 1   -2      1     1      -2 | tee -a result_verification_op2_var03.csv
        # rectangular matrices, a row major and b col major
	./run_test_op2_var03.x  ${MIN} ${MAX} ${STEP} 1 -${MIN}   1      -1     -1      1 | tee -a result_verification_op2_var03.csv
        # rectangular matrices, a row major and b col major
	./run_test_op2_var03.x  ${MIN} ${MAX} ${STEP} -${MIN} 1   1      -1     -1      1 | tee -a result_verification_op2_var03.csv
	grep -i "FAIL" result_verification_op2_var03.csv | wc -l



build_verifier:
	./build_test_op2.sh

# TODO: Build timer
build_bench:
	./build_bench_op2.sh

# TODO: Run Timer

clean:
	rm -f *.x *~ *.o

cleanall: clean
	rm -f *.csv
