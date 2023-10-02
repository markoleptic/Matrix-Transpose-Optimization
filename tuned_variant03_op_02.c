/* DANIEL YOWELL -

This variant implements OpenMP threading to parallelize multiple 8x8 transpositions.
2 THREADS

*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

#define MAT_SIDE_LENGTH 8

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

/** An array of 8 __m256 (64 elements) */
struct Matrix_64
{
  __m256 rows[MAT_SIDE_LENGTH];
};
typedef struct Matrix_64 Matrix_64;

/** Loads 64 elements from the source array into a Matrix_64 struct. The stride
 *  specifies how far apart the elements of each row are in memory. For example,
 *  a stride of 1 would mean that the elements are contiguous in memory. */
void loadMat64(Matrix_64 *mat, const float *source, size_t stride)
{
  for (int i = 0; i < MAT_SIDE_LENGTH; i++)
  {
    mat->rows[i] = _mm256_loadu_ps(source + stride * i);
  }
}

/** Stores 64 elements from the Matrix_64 struct into a destination array. Same
 *  stride parameter as loadMat64. */
void storeMat64(const Matrix_64 *mat, float *dest, size_t stride)
{
  for (int i = 0; i < MAT_SIDE_LENGTH; i++)
  {
    _mm256_storeu_ps(dest + stride * i, mat->rows[i]);
  }
}

/** In place transpose of an 8x8 matrix
 *
 * Example unpacklo and unpackhi:
 * [00, 01, 02, 03, 04, 05, 06, 07]
 * [08, 09, 10, 11, 12, 13, 14, 15]
 *
 * [00, 08, 01, 09, 02, 10, 03, 11]
 * [04, 12, 05, 13, 06, 14, 07, 15]
 *
 * _MM_SHUFFLE(1, 0, 1, 0):
 * 1st & 2nd from tmp0,
 * 1st & 2nd from tmp2,
 * 5th & 6th from tmp0,
 * 5th & 6th from tmp2.
 *
 * _MM_SHUFFLE(3, 2, 3, 2):
 * 3rd & 4th from tmp0,
 * 3rd & 4th from tmp2,
 * 7th & 8th from tmp0,
 * 7th & 8th from tmp2.
 *
 * _mm256_permute2f128_ps(shuff0, shuff4, 0x20):
 * 1st half of shuff0,
 * 1st half of shuff4.
 *
 * _mm256_permute2f128_ps(shuff0, shuff4, 0x31):
 * 2nd half of shuff0,
 * 2nd half of shuff4.
 */
void transposeMat64(Matrix_64 *m8)
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

/** Transposes a larger matrix by tiling into 8x8 blocks (Matrix_64) */
void transposeLargeMat(int m, int n, float *src, int rs_s, int cs_s, float *dst, int rs_d, int cs_d)
{
  int numBlocksM = m / MAT_SIDE_LENGTH;
  int numBlocksN = n / MAT_SIDE_LENGTH;
  // Loop through each 8x8 block
  #pragma omp parallel for num_threads(2)
  for (int i = 0; i < numBlocksM; i++)
  {
    for (int j = 0; j < numBlocksN; j++)
    {
      Matrix_64 block;
      /** This handles when strides are (2, 16, 16, 2), etc. Main difference from the rest is that
       *  MAT_SIDE_LENGTH is subtracted from both, otherwise the last 8 elements of the 8x8 block
       *  will be incorrect. It also doesn't transpose. */
      if (rs_s > 1 && rs_d > 1 && cs_s > 1 && cs_d > 1)
      {
        loadMat64(&block, src + (i * MAT_SIDE_LENGTH) + (j * MAT_SIDE_LENGTH * cs_s) - MAT_SIDE_LENGTH, cs_s);
        storeMat64(&block, dst + (i * MAT_SIDE_LENGTH) + (j * MAT_SIDE_LENGTH * cs_s) - MAT_SIDE_LENGTH, cs_s);
      }
      /** This handles when strides are ( 16, 1, 16, 1), etc. The loading and saving are the same
       *  except for i and j are swapped. Same pattern as the next one. */
      else if (cs_s == 1 && cs_d == 1)
      {
        loadMat64(&block, src + (i * MAT_SIDE_LENGTH * rs_s) + (j * MAT_SIDE_LENGTH), rs_s);
        transposeMat64(&block);
        storeMat64(&block, dst + (j * MAT_SIDE_LENGTH * rs_s) + (i * MAT_SIDE_LENGTH), rs_s);
      }
      /** This handles when strides are (1, 16, 1, 16), etc. The loading and saving are the same
       *  except for i and j are swapped. Same pattern as the previous one.*/
      else if (rs_s == 1 && rs_d == 1)
      {
        loadMat64(&block, src + (i * MAT_SIDE_LENGTH) + (j * MAT_SIDE_LENGTH * cs_s), cs_s);
        transposeMat64(&block);
        storeMat64(&block, dst + (j * MAT_SIDE_LENGTH) + (i * MAT_SIDE_LENGTH * cs_s), cs_s);
      }
      /** This handles when strides are (16, 1, 1, 16), etc. No transpose needed and same pattern
       *  as the next one. */
      else if (cs_s == 1 && rs_d == 1)
      {
        loadMat64(&block, src + (i * MAT_SIDE_LENGTH * rs_s) + (j * MAT_SIDE_LENGTH), rs_s);
        storeMat64(&block, dst + (j * MAT_SIDE_LENGTH) + (i * MAT_SIDE_LENGTH * rs_s), rs_s);
      }
      /** This handles when strides are (1, 16, 16, 1), etc. No transpose needed and same pattern
       *  as the previous one. */
      else if (rs_s == 1 && cs_d == 1)
      {
        loadMat64(&block, src + (i * MAT_SIDE_LENGTH * cs_s) + (j * MAT_SIDE_LENGTH), cs_s);
        storeMat64(&block, dst + (j * MAT_SIDE_LENGTH) + (i * MAT_SIDE_LENGTH * cs_s), cs_s);
      }
    }
  }
}

#ifndef FUN_NAME
#define FUN_NAME baseline_transpose
#endif

void FUN_NAME(int m, int n, float *src, int rs_s, int cs_s, float *dst, int rs_d, int cs_d)
{
  transposeLargeMat(m, n, src, rs_s, cs_s, dst, rs_d, cs_d);
}