#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Reference function
#ifndef FUN_NAME_REF
#define FUN_NAME_REF baseline_transpose
#endif

// Function under test
#ifndef FUN_NAME_TST
#define FUN_NAME_TST baseline_transpose
#endif


extern void FUN_NAME_REF( int m, int n,
			  float *src,
			  int rs_s, int cs_s,
			  float *dst,
			  int rs_d, int cs_d);

extern void FUN_NAME_TST( int m, int n,
			  float *src,
			  int rs_s, int cs_s,
			  float *dst,
			  int rs_d, int cs_d);



void fill_buffer_with_random( int num_elems, float *buff )
{
    for(int i = 0; i < num_elems; ++i)
      {
	buff[i] = ((float)(rand()-((RAND_MAX)/2)))/((float)RAND_MAX);
      }
}


float max_pair_wise_diff(int m, int n, int rs, int cs, float *a, float *b)
{
  float max_diff = 0.0;
  
  for(int i = 0; i < m; ++i)
      for(int j = 0; j < n; ++j)
    {
      float sum  = fabs(a[i*rs+j*cs]+b[i*rs+j*cs]);
      float diff = fabs(a[i*rs+j*cs]-b[i*rs+j*cs]);

      float res = 0.0f;

      if(sum == 0.0f)
	res = diff;
      else
	res = 2*diff/sum;

      if( res > max_diff )
	max_diff = res;
    }

  return max_diff;
}


int scale_p_on_pos_ret_v_on_neg(int p, int v)
{
  if (v < 1)
    return -1*v;
  else
    return v*p;
}


int main( int argc, char *argv[] )
{

  //       or use the defaults.
  int min_size;
  int max_size;
  int step_size;
  // defaults both row major
  int in_m;
  int in_n;
  
  int in_rs_src;
  int in_cs_src;
     
  int in_rs_dst;
  int in_cs_dst;

  if(argc == 1 )
    {
      min_size  = 16;
      max_size  = 256;
      step_size = 16;
      // defaults both row major
      in_m=1;
      in_n=1;
  
      in_rs_src=1;
      in_cs_src=-1;
     
      in_rs_dst=1;
      in_cs_dst=-1;

    }
  else if(argc == 9 + 1 )
    {
      min_size  = atoi(argv[1]);
      max_size  = atoi(argv[2]);
      step_size = atoi(argv[3]);

      in_m=atoi(argv[4]);
      in_n=atoi(argv[5]);
	 
      in_rs_src=atoi(argv[6]);
      in_cs_src=atoi(argv[7]);
     
      in_rs_dst=atoi(argv[8]);
      in_cs_dst=atoi(argv[9]);

    }
  else
    {
      printf("usage: %s min max step m n rs_src cs_src rs_dst cs_dst\n"
	     "\n"
	     "EXAMPLES:"
	     "row major for src and dst: 8 32 8 1 1 1 -1 1 -1\n"
	     "col major for src and dst: 8 32 8 1 1 -1 1 -1 1\n"
	     "",
	     argv[0]);
      exit(1);
    }

  // Print out the first line of the output in csv format
  printf("m, n, rs_src, cs_src, rs_dst, cs_dst, result\n");
 

  // src:Row Major Layout dst:Row Major Layout
 for( int p = min_size;
      p < max_size;
      p += step_size )
   {

#if 0 // manual sizes
     // matrix size
     int m=p;
     int n=p;

     // row major for src
     int rs_src = n;
     int cs_src = 1;

     // row major for dst
     int rs_dst = m; // Flipped from n to m because of transpose
     int cs_dst = 1;
#else
     
     // matrix size
     int m=scale_p_on_pos_ret_v_on_neg(p,in_m);
     int n=scale_p_on_pos_ret_v_on_neg(p,in_n);

     // row major for src
     int rs_src = scale_p_on_pos_ret_v_on_neg(n,in_rs_src);
     int cs_src = scale_p_on_pos_ret_v_on_neg(m,in_cs_src);

     // row major for dst
     // Flipped from n to m because of transpose
     int rs_dst = scale_p_on_pos_ret_v_on_neg(m,in_rs_dst);
     int cs_dst = scale_p_on_pos_ret_v_on_neg(n,in_cs_dst);

#endif

     // How big of a buffer do we need
     // This is overkill
     int buff_size_src=m*n*rs_src*cs_src;
     int buff_size_dst=m*n*rs_dst*cs_dst;
     
     float *src_ref = (float *)malloc(sizeof(float)*buff_size_src);
     float *dst_ref = (float *)malloc(sizeof(float)*buff_size_dst);

     float *src_tst = (float *)malloc(sizeof(float)*buff_size_src);
     float *dst_tst = (float *)malloc(sizeof(float)*buff_size_dst);

     // fill src_ref with random values
     fill_buffer_with_random( buff_size_src, src_ref );

     // copy src_ref to src_tst
     memcpy(src_tst,src_ref,buff_size_src*sizeof(float));

     // Run the reference
     FUN_NAME_REF( m, n,
		   src_ref,
		   rs_src, cs_src,
		   dst_ref,
		   rs_dst, cs_dst);


     // run the test
     FUN_NAME_TST( m, n,
		   src_tst,
		   rs_src, cs_src,
		   dst_tst,
		   rs_dst, cs_dst);



     float res = max_pair_wise_diff(n,m,rs_dst,cs_dst, dst_ref, dst_tst);

     printf("%i, %i, %i, %i, %i, %i, ",
	    m,n,
	    rs_src,cs_src,
	    rs_dst,cs_dst);

     // if our error is greater than some threshold
     if( res > 1e-6 )
       printf("FAIL\n");
     else
       printf("PASS\n");


     
     free(src_ref);
     free(dst_ref);
     free(src_tst);
     free(dst_tst);

     
   }
  
}
