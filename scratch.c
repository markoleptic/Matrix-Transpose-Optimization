#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>

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

struct Matrix_64
{
    __m256 rows[8];
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

void transposeLargeMat(int m, int n, float *src, int rs_s, int cs_s, float *dst, int rs_d, int cs_d)
{
    int numBlocksM = m / 8;
    int numBlocksN = n / 8;
    // Loop through each 8x8 block
    for (int i = 0; i < numBlocksM; i++)
    {
        for (int j = 0; j < numBlocksN; j++)
        {
            Matrix_64 block;
      if (rs_s > 1 && rs_d > 1 && cs_s > 1 && cs_d > 1)
      {
        loadMat64(&block, src + (i * MAT_SIDE_LENGTH) + (j * MAT_SIDE_LENGTH * cs_s) - MAT_SIDE_LENGTH, cs_s);
        storeMat64(&block, dst + (i * MAT_SIDE_LENGTH) + (j * MAT_SIDE_LENGTH * cs_s) - MAT_SIDE_LENGTH, cs_s);
      }
      else if (cs_s == 1 && cs_d == 1)
      {
        loadMat64(&block, src + (i * MAT_SIDE_LENGTH * rs_s) + (j * MAT_SIDE_LENGTH), rs_s);
        transposeMat64(&block);
        storeMat64(&block, dst + (j * MAT_SIDE_LENGTH * rs_s) + (i * MAT_SIDE_LENGTH), rs_s);
      }
      else if (rs_s == 1 && rs_d == 1)
      {
        loadMat64(&block, src + (i * MAT_SIDE_LENGTH) + (j * MAT_SIDE_LENGTH * cs_s), cs_s);
        transposeMat64(&block);
        storeMat64(&block, dst + (j * MAT_SIDE_LENGTH) + (i * MAT_SIDE_LENGTH * cs_s), cs_s);
      }
      else if (cs_s == 1 && rs_d == 1)
      {
        loadMat64(&block, src + (i * MAT_SIDE_LENGTH * rs_s) + (j * MAT_SIDE_LENGTH), rs_s);
        storeMat64(&block, dst + (j * MAT_SIDE_LENGTH) + (i * MAT_SIDE_LENGTH * rs_s), rs_s);
      }
      else if (rs_s == 1 && cs_d == 1)
      {
        loadMat64(&block, src + (i * MAT_SIDE_LENGTH * cs_s) + (j * MAT_SIDE_LENGTH), cs_s);
        storeMat64(&block, dst + (j * MAT_SIDE_LENGTH) + (i * MAT_SIDE_LENGTH * cs_s), cs_s);
      }
        }
    }
}

void copyMatrix(int M, int N, const float *source, float *dest)
{
    int numBlocksM = M / 8;
    int numBlocksN = N / 8;
    for (int i = 0; i < numBlocksM; i++)
    {
        for (int j = 0; j < numBlocksN; j++)
        {
            Matrix_64 block;
            loadMat64(&block, source + (i * 8 * N) + (j * 8), N);
            storeMat64(&block, dest + (i * 8 * N) + (j * 8), N);
        }
    }
}

void templateFunction(int m, int n, const float *src, int rs_s, int cs_s,
                      float *dst, int rs_d, int cs_d)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
        {
            dst[j * rs_d + i * cs_d] =
                src[i * rs_s + j * cs_s];
        }
}

// gcc -o scratch.o scratch.c -mavx
int main()
{
    int m = 16;
    int n = 16;
    int rs_s = 2;
    int cs_s = 16;
    int rs_d = 16;
    int cs_d = 2;
    int max_threads = omp_get_max_threads();
    int size = m * n; // The size of the array, including values from 0 to 255
    float *source = malloc(size * sizeof(float));
    float *dest_baseline = malloc(size * sizeof(float));
    float *dest_test = malloc(size * sizeof(float));

    for (int i = 0; i < size; i++)
    {
        source[i] = (float)i * 0.01f;
        dest_baseline[i] = 0.f;
        dest_test[i] = 0.f;
    }
    // baseline
    templateFunction(m, n, source, rs_s, cs_s, dest_baseline, rs_d, cs_d);
    // function in this file
    transposeLargeMat(m, n, source, rs_s, cs_s, dest_test, rs_d, cs_d);

    // copyMatrix(m,n,source,dest_test);
    // Print the transposed matrices
    printMatrix("out_baseline.txt", dest_baseline, 16, 16);
    printMatrix("out_test.txt", dest_test, 16, 16);

    free(source);
    free(dest_test);
    free(dest_baseline);
    return 0;
}

/* void transpose_big(int m, int n,
                   float *src,
                   int rs_s, int cs_s,
                   float *dst,
                   int rs_d, int cs_d)
{
    const char *filename = "numbers.txt";
    FILE *file = fopen(filename, "w");
    for (int i = 0; i < m; i += 4)
    {
        for (int j = 0; j < n; j += 4)
        {
            // Create a 4x4 block using Matrix_4x4
            Matrix_4x4 block;

            // Load the block from the source with custom strides
            fprintf(file, "src index: %d stride: %d\n", i * rs_s + j, rs_s);
            loadMat(&block, src + i * rs_s + j * cs_s, rs_s);

            // Transpose the block in-place
            transposeMat(&block);

            // Store the transposed block to the destination with custom strides
            storeMat(&block, dst + j * rs_d + i * cs_d, rs_d);
            fprintf(file, "dst index: %d stride: %d\n", j * rs_d + i * cs_d, rs_d);
        }
    }
}
 */

/* struct Matrix_4x4
{
    __m256 r0, r1, r2, r3;
};
typedef struct Matrix_4x4 Matrix_4x4;
void loadMat(Matrix_4x4 *mat, const float *source, size_t stride)
{
    mat->r0 = _mm256_loadu_ps(source);
    mat->r1 = _mm256_loadu_ps(source + stride);
    mat->r2 = _mm256_loadu_ps(source + stride * 2);
    mat->r3 = _mm256_loadu_ps(source + stride * 3);
}
void storeMat(const Matrix_4x4 *mat, float *dest, size_t stride)
{
    _mm256_storeu_ps(dest, mat->r0);
    _mm256_storeu_ps(dest + stride, mat->r1);
    _mm256_storeu_ps(dest + stride * 2, mat->r2);
    _mm256_storeu_ps(dest + stride * 3, mat->r3);
}
void transposeMat(Matrix_4x4 *m4)
{
    __m256 tmp0 = _mm256_unpacklo_ps(m4->r0, m4->r1);
    __m256 tmp1 = _mm256_unpackhi_ps(m4->r0, m4->r1);
    __m256 tmp2 = _mm256_unpacklo_ps(m4->r2, m4->r3);
    __m256 tmp3 = _mm256_unpackhi_ps(m4->r2, m4->r3);

    m4->r0 = _mm256_shuffle_ps(tmp0, tmp2, 0b01000100);
    m4->r1 = _mm256_shuffle_ps(tmp0, tmp2, 0b11101110);
    m4->r2 = _mm256_shuffle_ps(tmp1, tmp3, 0b01000100);
    m4->r3 = _mm256_shuffle_ps(tmp1, tmp3, 0b11101110);
}
 */

/* struct Matrix_4x4
{
  __m256 r0, r1, r2, r3;
};

typedef struct Matrix_4x4 Matrix_4x4;

void loadMat(Matrix_4x4 *mat, const float *source, size_t stride)
{
    mat->r0 = _mm256_loadu_ps(source);
    mat->r1 = _mm256_loadu_ps(source + stride);
    mat->r2 = _mm256_loadu_ps(source + stride * 2);
    mat->r3 = _mm256_loadu_ps(source + stride * 3);
}

void storeMat(const Matrix_4x4 *mat, float *dest, size_t stride)
{
  _mm256_storeu_ps(dest, mat->r0);
  _mm256_storeu_ps(dest + stride, mat->r1);
  _mm256_storeu_ps(dest + stride * 2, mat->r2);
  _mm256_storeu_ps(dest + stride * 3, mat->r3);
}

void transposeMat(Matrix_4x4 *m4)
{
  __m256 tmp0 = _mm256_unpacklo_ps(m4->r0, m4->r1);
  __m256 tmp1 = _mm256_unpackhi_ps(m4->r0, m4->r1);
  __m256 tmp2 = _mm256_unpacklo_ps(m4->r2, m4->r3);
  __m256 tmp3 = _mm256_unpackhi_ps(m4->r2, m4->r3);

  m4->r0 = _mm256_shuffle_ps(tmp0, tmp2, 0b01000100);
  m4->r1 = _mm256_shuffle_ps(tmp0, tmp2, 0b11101110);
  m4->r2 = _mm256_shuffle_ps(tmp1, tmp3, 0b01000100);
  m4->r3 = _mm256_shuffle_ps(tmp1, tmp3, 0b11101110);
}

struct Matrix_8x8
{
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;
};
typedef struct Matrix_8x8 Matrix_8x8;

void loadMat8x8(Matrix_8x8 *mat, const float *source, size_t stride)
{
    mat->r0 = _mm256_loadu_ps(source);
    mat->r1 = _mm256_loadu_ps(source + stride);
    mat->r2 = _mm256_loadu_ps(source + stride * 2);
    mat->r3 = _mm256_loadu_ps(source + stride * 3);
    mat->r4 = _mm256_loadu_ps(source + stride * 4);
    mat->r5 = _mm256_loadu_ps(source + stride * 5);
    mat->r6 = _mm256_loadu_ps(source + stride * 6);
    mat->r7 = _mm256_loadu_ps(source + stride * 7);
}

void storeMat8x8(const Matrix_8x8 *mat, float *dest, size_t stride)
{
    _mm256_storeu_ps(dest, mat->r0);
    _mm256_storeu_ps(dest + stride, mat->r1);
    _mm256_storeu_ps(dest + stride * 2, mat->r2);
    _mm256_storeu_ps(dest + stride * 3, mat->r3);
    _mm256_storeu_ps(dest + stride * 4, mat->r4);
    _mm256_storeu_ps(dest + stride * 5, mat->r5);
    _mm256_storeu_ps(dest + stride * 6, mat->r6);
    _mm256_storeu_ps(dest + stride * 7, mat->r7);
}

void transposeMat8x8(Matrix_8x8 *m8)
{
    __m256 tmp0 = _mm256_unpacklo_ps(m8->r0, m8->r1);
    __m256 tmp1 = _mm256_unpackhi_ps(m8->r0, m8->r1);
    __m256 tmp2 = _mm256_unpacklo_ps(m8->r2, m8->r3);
    __m256 tmp3 = _mm256_unpackhi_ps(m8->r2, m8->r3);
    __m256 tmp4 = _mm256_unpacklo_ps(m8->r4, m8->r5);
    __m256 tmp5 = _mm256_unpackhi_ps(m8->r4, m8->r5);
    __m256 tmp6 = _mm256_unpacklo_ps(m8->r6, m8->r7);
    __m256 tmp7 = _mm256_unpackhi_ps(m8->r6, m8->r7);

    __m256 __tt0 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(1,0,1,0));
    __m256 __tt1 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3,2,3,2));
    __m256 __tt2 = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(1,0,1,0));
    __m256 __tt3 = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3,2,3,2));
    __m256 __tt4 = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(1,0,1,0));
    __m256 __tt5 = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(3,2,3,2));
    __m256 __tt6 = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(1,0,1,0));
    __m256 __tt7 = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(3,2,3,2));

    m8->r0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    m8->r1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    m8->r2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    m8->r3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    m8->r4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    m8->r5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    m8->r6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    m8->r7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

void transpose_big(int m, int n,
                       float *src,
                       int rs_s, int cs_s,
                       float *dst,
                       int rs_d, int cs_d)
{
  for (size_t i = 0; i < m; i += 4)
  {
    for (size_t j = 0; j < n; j += 4)
    {
      // Create a 4x4 block using Matrix_4x4
      Matrix_4x4 block;

      // Load the block from the source with custom strides
      loadMat(&block, src + i * rs_s + j * cs_s, rs_s);

      // Transpose the block in-place
      transposeMat(&block);

      // Store the transposed block to the destination with custom strides
      storeMat(&block, dst + j * rs_d + i * cs_d, rs_d);
    }
  }
} */

/*     __m256 tmp0 = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    __m256 tmp2 = _mm256_setr_ps(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

    // Shuffle tmp0 and tmp2 using control masks
   __m256 __tt0 = _mm256_permute2f128_ps(tmp0, tmp2, 0x20);
   __m256 __tt1 = _mm256_permute2f128_ps(tmp0, tmp2, 0x31);

    // Extract and print the results
    float result_tt0[8];
    float result_tt1[8];
    _mm256_storeu_ps(result_tt0, __tt0);
    _mm256_storeu_ps(result_tt1, __tt1);

    printf("Result for __tt0: ");
    for (int i = 0; i < 8; i++) {
        printf("%.1f ", result_tt0[i]);
    }
    printf("\n");

    printf("Result for __tt1: ");
    for (int i = 0; i < 8; i++) {
        printf("%.1f ", result_tt1[i]);
    }
    printf("\n"); */