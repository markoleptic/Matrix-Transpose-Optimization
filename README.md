# Matrix Transposition Optimization

This repository implements various parallel optimization techniques for matrix transposition. The starting point for all variants is `baseline_op_02.c`.

## Optimization Techniques

- **tuned_variant01_op_02:** Utilizes AVX2 to parallelize the matrix into sub-blocks.
- **tuned_variant02_op_02:** Modifies the transposition function based on the row and column strides.
- **tuned_variant03_op_02:** OpenMP 2 threads.
- **tuned_variant04_op_02:** OpenMP 3 threads.
- **tuned_variant05_op_02:** Applies loop unrolling.
- **tuned_variant06_op_02:** Combines the best of the previous techniques.

## File Descriptions

- **baseline_op_02.c:** The starting point for all variants.
- **writeup.pdf:** Contains a full description and results of the implemented optimization techniques.

Feel free to explore each file for detailed implementation and optimization strategies.

