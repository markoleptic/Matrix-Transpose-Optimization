/*

  m,n >= 0:   dimension of src matrix
  m: number of rows
  n: number of columns

  float* src: source matrix (m-by-n matrix)
  rs_s, cs_s >= 1: row and column stride of source matrix
  rs_s: distance in memory between rows (rs_s = 1 --> column major ordering)
  cs_s: distance in memory between columns (cs_s = 1 --> row major ordering)

  float* dst: destination matrix (n-by-m matrix)
  rs_d, cs_d >= 1: row and column stride of destination matix

  NOTE: This is an out-of-place transposition meaning src and
        dst WILL NOT OVERLAP.

*/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef FUN_NAME
#define FUN_NAME baseline_transpose
#endif

void FUN_NAME( int m, int n,
		float *src,
		int rs_s, int cs_s,
		float *dst,
		int rs_d, int cs_d)
{
  for( int i = 0; i < m; ++i )
    for( int j = 0; j < n; ++j )
      {
	dst[ j*rs_d + i*cs_d ] =
	  src[ i*rs_s + j*cs_s ];
      }
}

/* if (m == 32 && n ==32 && rs_s == 2) {
      FILE *file = fopen("out_baseline.txt", "w");
      if (file == NULL)
      {
          perror("Failed to open file");
          return;
      }
      for (int i = 0; i < m; i++)
      {
          for (int j = 0; j < n; j++)
          {
              fprintf(file, "%8.2f ", dst[i * m + j]);
          }
          fprintf(file, "\n");
      }
      fclose(file);
} */