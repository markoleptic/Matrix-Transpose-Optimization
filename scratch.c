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

void transposeLargetMat(float *source, float *dest, int M, int N) {
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

int main()
{
    float source[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 
                    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 
                    16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 
                    24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 
                    32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 
                    40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 
                    48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 
                    56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f,
                    64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 
                    72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 
                    80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 
                    88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0,
                    96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 
                    104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 
                    112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 
                    120.0,121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0};
    float *src = source;
    float *dest = malloc(128 * sizeof(float));

    transposeLargetMat(source, dest, 16, 8);

    // Print the transposed matrix
    const char *filename = "out.txt";
    printMatrix(filename, dest, 8, 16);

    free(dest);

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
