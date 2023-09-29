#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>


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

struct Matrix_64
{
    __m256 rows[8];

    // Row index within the larger matrix
    int rowIndex;

    // Column index within the larger matrix
    int colIndex;
};
typedef struct Matrix_64 Matrix_64;

void loadMat64(Matrix_64 *mat, const float *source, size_t stride)
{
    mat->rows[0] = _mm256_loadu_ps(source);
    mat->rows[1] = _mm256_loadu_ps(source + stride);
    mat->rows[2] = _mm256_loadu_ps(source + stride * 2);
    mat->rows[3] = _mm256_loadu_ps(source + stride * 3);
    mat->rows[4] = _mm256_loadu_ps(source + stride * 4);
    mat->rows[5] = _mm256_loadu_ps(source + stride * 5);
    mat->rows[6] = _mm256_loadu_ps(source + stride * 6);
    mat->rows[7] = _mm256_loadu_ps(source + stride * 7);
}

void storeMat64(const Matrix_64 *mat, float *dest, size_t stride)
{
    _mm256_storeu_ps(dest, mat->rows[0]);
    _mm256_storeu_ps(dest + stride, mat->rows[1]);
    _mm256_storeu_ps(dest + stride * 2, mat->rows[2]);
    _mm256_storeu_ps(dest + stride * 3, mat->rows[3]);
    _mm256_storeu_ps(dest + stride * 4, mat->rows[4]);
    _mm256_storeu_ps(dest + stride * 5, mat->rows[5]);
    _mm256_storeu_ps(dest + stride * 6, mat->rows[6]);
    _mm256_storeu_ps(dest + stride * 7, mat->rows[7]);
}

void transposeMat64(Matrix_64 *m8)
{
    __m256 tmp0 = _mm256_unpacklo_ps(m8->rows[0], m8->rows[1]);
    __m256 tmp1 = _mm256_unpackhi_ps(m8->rows[0], m8->rows[1]);
    __m256 tmp2 = _mm256_unpacklo_ps(m8->rows[2], m8->rows[3]);
    __m256 tmp3 = _mm256_unpackhi_ps(m8->rows[2], m8->rows[3]);
    __m256 tmp4 = _mm256_unpacklo_ps(m8->rows[4], m8->rows[5]);
    __m256 tmp5 = _mm256_unpackhi_ps(m8->rows[4], m8->rows[5]);
    __m256 tmp6 = _mm256_unpacklo_ps(m8->rows[6], m8->rows[7]);
    __m256 tmp7 = _mm256_unpackhi_ps(m8->rows[6], m8->rows[7]);

    __m256 __tt0 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 __tt1 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 __tt2 = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 __tt3 = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 __tt4 = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 __tt5 = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 __tt6 = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 __tt7 = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(3, 2, 3, 2));

    m8->rows[0] = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    m8->rows[1] = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    m8->rows[2] = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    m8->rows[3] = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    m8->rows[4] = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    m8->rows[5] = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    m8->rows[6] = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    m8->rows[7] = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

void transposeLargeMat(float *source, float *dest, int M, int N) {
    int numBlocksM = M / 8;
    int numBlocksN = N / 8;
    // Loop through each 8x8 block
    for (int i = 0; i < numBlocksM; i++) {
        for (int j = 0; j < numBlocksN; j ++) {
            // Create a Matrix_64 block for the current 8x8 block
            Matrix_64 block;
            loadMat64(&block, source + i * 8 * N + j * 8, N);

            // Transpose the block in-place
            transposeMat64(&block);

            // Store the transposed block to the destination
            storeMat64(&block, dest + j * 8 * M + i * 8, M);
        }
    }
}


#ifndef FUN_NAME
#define FUN_NAME baseline_transpose
#endif

void FUN_NAME(int m, int n, float *src, int rs_s, int cs_s, float *dst, int rs_d, int cs_d)
{
  transposeLargeMat(src, dst, m, n);
}

// for (int i = 0; i < m; ++i)
// {
//   for (int j = 0; j < n; ++j)
//   {
//     dst[j * rs_d + i * cs_d] = src[i * rs_s + j * cs_s];
//   }
// } 

/*   printf("src: \n\n");
  for (int i = 0; i < m; i ++)
  {
    for (int j = 0; j < n; j ++)
    {
      printf("%f ", src[i*j + j]);
    }
    printf("\n");
  }
  M256_Set *block = malloc(sizeof(M256_Set));
  // Top-left corner
  loadMat(block, src, 8);
  transposeMat(block);
  storeMat(block, dst, 8);

  // Using another instance of the block to support in-place transpose, with very small overhead
  M256_Set *block2 = malloc(sizeof(M256_Set));
  loadMat(block, src + 4, 8);      // top right block
  loadMat(block2, src + 8 * 4, 8); // bottom left block

  transposeMat(block2);
  storeMat(block2, dst, 8);
  transposeMat(block);
  storeMat(block, dst, 8);

  // Bottom-right corner
  loadMat(block, src , 8);
  transposeMat(block);
  storeMat(block, dst, 8);

  free(block);
  free(block2); */
/*   printf("dst: \n\n");
  for (int i = 0; i < m; i ++)
  {
    for (int j = 0; j < n; j ++)
    {
      printf("%f ", dst[i*j + j]);
    }
    printf("\n");
  } */
/*   for (int i = 0; i < m; i += 8)
  {
    for (int j = 0; j < n; j += 8)
    {
      dst[j * rs_d + i * cs_d] = src[i * rs_s + j * cs_s];
      __m256 input00_07 = _mm256_loadu_ps(&src[i * rs_s + j * cs_s]);
      __m256 input08_15 = _mm256_loadu_ps(&src[(i * rs_s + j * cs_s) + 8]);
      __m256 input16_23 = _mm256_loadu_ps(&src[(i * rs_s + j * cs_s) + 8 + 8]);
      __m256 input24_31 = _mm256_loadu_ps(&src[(i * rs_s + j * cs_s) + 8 + 8 + 8]);
      __m256 input32_39 = _mm256_loadu_ps(&src[(i * rs_s + j * cs_s) + 8 + 8 + 8 + 8]);
      __m256 input40_47 = _mm256_loadu_ps(&src[(i * rs_s + j * cs_s) + 8 + 8 + 8 + 8 + 8]);
      __m256 input48_55 = _mm256_loadu_ps(&src[(i * rs_s + j * cs_s) + 8 + 8 + 8 + 8 + 8 + 8]);
      __m256 input56_63 = _mm256_loadu_ps(&src[(i * rs_s + j * cs_s) + 8 + 8 + 8 + 8 + 8 + 8 + 8]);

      __m256 __t0 = _mm256_unpacklo_ps(input00_07, input08_15);
      __m256 __t1 = _mm256_unpackhi_ps(input00_07, input08_15);
      __m256 __t2 = _mm256_unpacklo_ps(input16_23, input24_31);
      __m256 __t3 = _mm256_unpackhi_ps(input16_23, input24_31);
      __m256 __t4 = _mm256_unpacklo_ps(input32_39, input40_47);
      __m256 __t5 = _mm256_unpackhi_ps(input32_39, input40_47);
      __m256 __t6 = _mm256_unpacklo_ps(input48_55, input56_63);
      __m256 __t7 = _mm256_unpackhi_ps(input48_55, input56_63);

      __m256 __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
      __m256 __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
      __m256 __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
      __m256 __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
      __m256 __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
      __m256 __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
      __m256 __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
      __m256 __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));

      __m256 output00_07 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
      __m256 output08_15 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
      __m256 output16_23 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
      __m256 output24_31 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
      __m256 output32_39 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
      __m256 output40_47 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
      __m256 output48_55 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
      __m256 output56_63 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);

      // Store the result back to memory.
      _mm256_storeu_ps(&dst[j * rs_d + i * cs_d], output00_07);
      _mm256_storeu_ps(&dst[(j * rs_d + i * cs_d) + 8], output08_15);
      _mm256_storeu_ps(&dst[(j * rs_d + i * cs_d) + 8 + 8], output16_23);
      _mm256_storeu_ps(&dst[(j * rs_d + i * cs_d) + 8 + 8 + 8], output24_31);
      _mm256_storeu_ps(&dst[(j * rs_d + i * cs_d) + 8 + 8 + 8 + 8], output32_39);
      _mm256_storeu_ps(&dst[(j * rs_d + i * cs_d) + 8 + 8 + 8 + 8 + 8], output40_47);
      _mm256_storeu_ps(&dst[(j * rs_d + i * cs_d) + 8 + 8 + 8 + 8 + 8 + 8], output48_55);
      _mm256_storeu_ps(&dst[(j * rs_d + i * cs_d) + 8 + 8 + 8 + 8 + 8 + 8 + 8], output56_63);
     }
  }*/
/*   for (int i = 0; i < m; i += 8)
  {
    for (int j = 0; j < n; j += 8)
    {
      // transpose the block beginning at [i,j]
      for (int k = i; k < i + 8; ++k)
      {
        for (int l = j; l < j + 8; ++l)
        {

          //dst[l * rs_d + k * cs_d] = src[k * rs_s + l * cs_s];
        }
      }
    }
  } */