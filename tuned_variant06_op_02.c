/** Variant 6, combines all of the previous variants, and dynamically controls the number of threads. */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>

/** Prints matrix to output file. */
void printMatrix(const char *filename, const float *matrix, int rows, int cols)
{
  FILE *file = fopen(filename, "w");
  if (file == NULL)
  {
    perror("Failed to open file");
    return;
  }
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      fprintf(file, "%8.2f ", matrix[i * cols + j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
}

#define MAT_SIDE_LENGTH 8
#define MAX_THREADS 4

/** An array of 8 __m256 (64 elements). */
struct Matrix_64
{
  __m256 rows[MAT_SIDE_LENGTH] __attribute__((aligned(32)));
};
typedef struct Matrix_64 Matrix_64;

/** Function type for performing a transpose */
typedef void (*TransposeFunction)(float *, float *, int, int, int, int);

/** Function type for looping through a larger matrix */
typedef void (*ParallelFunction)(TransposeFunction transposeFunc, int m, int n, float *src, int rs_s, int cs_s, float *dst);

/** Loads 64 elements from the source array into a Matrix_64 struct. */
inline void loadMat64(Matrix_64 *mat, const float *source, size_t stride)
{
  for (int i = 0; i < MAT_SIDE_LENGTH; i+=2)
  {
    mat->rows[i] = _mm256_loadu_ps(source + stride * i);
    mat->rows[i+1] = _mm256_loadu_ps(source + stride * (i+1) );
  }
}

/** Stores 64 elements from the Matrix_64 struct into a destination array. */
inline void storeMat64(const Matrix_64 *mat, float *dest, size_t stride)
{
  for (int i = 0; i < MAT_SIDE_LENGTH; i+=2)
  {
    _mm256_storeu_ps(dest + stride * i, mat->rows[i]);
    _mm256_storeu_ps(dest + stride * (i+1), mat->rows[(i+1)]);
  }
}

/** In place transpose of an 8x8 matrix. */
inline void transposeMat64(Matrix_64 *m8)
{
  // Unpack and interleave rows of the matrix
  __m256 tmp0 = _mm256_unpacklo_ps(m8->rows[0], m8->rows[1]);
  __m256 tmp1 = _mm256_unpackhi_ps(m8->rows[0], m8->rows[1]);
  __m256 tmp2 = _mm256_unpacklo_ps(m8->rows[2], m8->rows[3]);
  __m256 tmp3 = _mm256_unpackhi_ps(m8->rows[2], m8->rows[3]);
  __m256 tmp4 = _mm256_unpacklo_ps(m8->rows[4], m8->rows[5]);
  __m256 tmp5 = _mm256_unpackhi_ps(m8->rows[4], m8->rows[5]);
  __m256 tmp6 = _mm256_unpacklo_ps(m8->rows[6], m8->rows[7]);
  __m256 tmp7 = _mm256_unpackhi_ps(m8->rows[6], m8->rows[7]);

  // more interleaving
  __m256 shuff0 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 shuff1 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3, 2, 3, 2));
  __m256 shuff2 = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 shuff3 = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3, 2, 3, 2));
  __m256 shuff4 = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 shuff5 = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(3, 2, 3, 2));
  __m256 shuff6 = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 shuff7 = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(3, 2, 3, 2));

  // Final swap and store
  m8->rows[0] = _mm256_permute2f128_ps(shuff0, shuff4, 0x20);
  m8->rows[1] = _mm256_permute2f128_ps(shuff1, shuff5, 0x20);
  m8->rows[2] = _mm256_permute2f128_ps(shuff2, shuff6, 0x20);
  m8->rows[3] = _mm256_permute2f128_ps(shuff3, shuff7, 0x20);
  m8->rows[4] = _mm256_permute2f128_ps(shuff0, shuff4, 0x31);
  m8->rows[5] = _mm256_permute2f128_ps(shuff1, shuff5, 0x31);
  m8->rows[6] = _mm256_permute2f128_ps(shuff2, shuff6, 0x31);
  m8->rows[7] = _mm256_permute2f128_ps(shuff3, shuff7, 0x31);
}

/** This handles when strides are (2, 16, 16, 2), etc. */
void transposeRS_RS_CS_CS(float *src, float *dst, int rs_s, int cs_s, int i, int j)
{
  Matrix_64 block;
  loadMat64(&block, src + (i * MAT_SIDE_LENGTH) + (j * MAT_SIDE_LENGTH * cs_s) - MAT_SIDE_LENGTH, cs_s);
  storeMat64(&block, dst + (i * MAT_SIDE_LENGTH) + (j * MAT_SIDE_LENGTH * cs_s) - MAT_SIDE_LENGTH, cs_s);
}

/** This handles when strides are ( 16, 1, 16, 1), etc (Row-Major storage). */
void transposeCS_CS(float *src, float *dst, int rs_s, int cs_s, int i, int j)
{
  Matrix_64 block;
  loadMat64(&block, src + (i * MAT_SIDE_LENGTH * rs_s) + (j * MAT_SIDE_LENGTH), rs_s);
  transposeMat64(&block);
  storeMat64(&block, dst + (j * MAT_SIDE_LENGTH * rs_s) + (i * MAT_SIDE_LENGTH), rs_s);
}

/** This handles when strides are (1, 16, 1, 16), etc (Column-Major storage). */
void transposeRS_RS(float *src, float *dst, int rs_s, int cs_s, int i, int j)
{
  Matrix_64 block;
  loadMat64(&block, src + (i * MAT_SIDE_LENGTH) + (j * MAT_SIDE_LENGTH * cs_s), cs_s);
  transposeMat64(&block);
  storeMat64(&block, dst + (j * MAT_SIDE_LENGTH) + (i * MAT_SIDE_LENGTH * cs_s), cs_s);
}

/** This handles when strides are (16, 1, 1, 16), etc. */
void transposeCS_RS(float *src, float *dst, int rs_s, int cs_s, int i, int j)
{
  Matrix_64 block;
  loadMat64(&block, src + (i * MAT_SIDE_LENGTH * rs_s) + (j * MAT_SIDE_LENGTH), rs_s);
  storeMat64(&block, dst + (j * MAT_SIDE_LENGTH) + (i * MAT_SIDE_LENGTH * rs_s), rs_s);
}

/** This handles when strides are (1, 16, 16, 1), etc.  */
void transposeRS_CS(float *src, float *dst, int rs_s, int cs_s, int i, int j)
{
  Matrix_64 block;
  loadMat64(&block, src + (i * MAT_SIDE_LENGTH * cs_s) + (j * MAT_SIDE_LENGTH), cs_s);
  storeMat64(&block, dst + (j * MAT_SIDE_LENGTH) + (i * MAT_SIDE_LENGTH * cs_s), cs_s);
}

/** Returns the appropriate TransposeFunction depending on the row stride and column stride. */
inline TransposeFunction selectTransposeFunction(int rs_s, int cs_s, int rs_d, int cs_d)
{
  if (cs_s == 1)
  {
    return cs_d == 1 ? transposeCS_CS : transposeCS_RS;
  }
  else if (rs_s == 1)
  {
    return rs_d == 1 ? transposeRS_RS : transposeRS_CS;
  }
  return transposeRS_RS_CS_CS;
}

/** Transposes a larger matrix by tiling into 8x8 blocks (Matrix_64), using the TransposeFunction at each block. */
void transposeLargeMat_NoThreading(TransposeFunction transposeFunc, int m, int n, float *src, int rs_s, int cs_s, float *dst)
{
  // Loop through each 8x8 block
  for (int i = 0; i < m / MAT_SIDE_LENGTH; i++)
  {
    for (int j = 0; j < n / MAT_SIDE_LENGTH; j++)
    {
      transposeFunc(src, dst, rs_s, cs_s, i, j);
    }
  }
}

/** Transposes a larger matrix by tiling into 8x8 blocks (Matrix_64), using the TransposeFunction at each block. */
void transposeLargeMat_DynamicThreads(TransposeFunction transposeFunc, int m, int n, float *src, int rs_s, int cs_s, float *dst)
{
  int sys_threads = omp_get_max_threads();
  // One thread for each outer loop
  int ideal_threads = m / MAT_SIDE_LENGTH;
  int threads = ideal_threads <= sys_threads ? ideal_threads : sys_threads;
  // Place hard cap on number of threads bc managing threads is also resource intensive
  threads = threads <= MAX_THREADS ? threads : MAX_THREADS;
  #pragma omp parallel for num_threads(threads)
  for (int i = 0; i < m / MAT_SIDE_LENGTH; i++)
  {
    for (int j = 0; j < n / MAT_SIDE_LENGTH; j++)
    {
      transposeFunc(src, dst, rs_s, cs_s, i, j);
    }
  }
}

/** Selects appropriate function based on size of matrix and stride */
inline ParallelFunction selectParallelFunction(int m, int n, int rs_s, int cs_s, int rs_d, int cs_d) {
  if ((m == n) && (m > 64) && ((cs_s == rs_d == 1) || rs_s == cs_d == 1)) {
    return transposeLargeMat_DynamicThreads;
  }
  return transposeLargeMat_NoThreading;
}

#ifndef FUN_NAME
#define FUN_NAME baseline_transpose
#endif

void FUN_NAME(int m, int n, float *src, int rs_s, int cs_s, float *dst, int rs_d, int cs_d)
{
  ParallelFunction Function = selectParallelFunction(m, n, rs_s, cs_s, rs_d, cs_d);
  Function(selectTransposeFunction(rs_s, cs_s, rs_d, cs_d), m, n, src, rs_s, cs_s, dst);
}