/*
  Timer harness for running a "function under test" for num_runs number of
  runs.


*/
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

#include "timer.h"

// Function under test
#ifndef FUN_NAME_TST
#define FUN_NAME_TST baseline_transpose
#endif

extern void FUN_NAME_TST( int m, int n,
			  float *src,
			  int rs_s, int cs_s,
			  float *dst,
			  int rs_d, int cs_d);


long pick_min_in_list(int num_trials, long *results)
{
  long current_min = LONG_MAX;

  for( int i = 0; i < num_trials; ++i )
    if( results[i] < current_min )
      current_min = results[i];

  return current_min;
}

void flush_cache()
{
  
  int size = 1024*1024*8;

  int *buff = (int *)malloc(sizeof(int)*size);
  int i, result = 0;
  volatile int sink;
  for (i = 0; i < size; i ++)
    result += buff[i];
  sink = result; /* So the compiler doesn't optimize away the loop */

  free(buff);
}

void time_function_under_test(int num_trials,
			      int num_runs_per_trial,
			      long *results, // results from each trial

			      int m, int n,
			      float *src,
			      int rs_s, int cs_s,
			      float *dst,
			      int rs_d, int cs_d
			      )
{
  // Initialize the start and stop variables.
  TIMER_INIT_COUNTERS(stop, start);

  // Click the timer a few times so the subsequent measurements are more accurate
  TIMER_WARMUP(stop,start);

  // flush the cache
  flush_cache();
  for(int trial = 0; trial < num_trials; ++trial )
    {

	/*
	  Time code.
	*/
        // start timer
	TIMER_GET_CLOCK(start);

	////////////////////////
        // Benchmark the code //
	////////////////////////

	for(int runs = 0; runs < num_runs_per_trial; ++runs )
	  {
	    FUN_NAME_TST( m, n,
			  src,
			  rs_s, cs_s,
			  dst,
			  rs_d, cs_d);
	  }

	////////////////////////
        // End Benchmark      //
	////////////////////////

        
        // stop timer
	TIMER_GET_CLOCK(stop);

	// subtract the start time from the stop time
	TIMER_GET_DIFF(start,stop,results[trial])

    }

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

  int num_trials = 30;
  int num_runs_per_trial = 30;


  ////////////////////////////////////


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
  printf("m, n, rs_src, cs_src, rs_dst, cs_dst, GB_per_s\n");
 

  // src:Row Major Layout dst:Row Major Layout
 for( int p = min_size;
      p < max_size;
      p += step_size )
   {
     
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

     
     // How big of a buffer do we need
     // This is overkill
     int buff_size_src=m*n*rs_src*cs_src;
     int buff_size_dst=m*n*rs_dst*cs_dst;
     
     float *src_tst = (float *)malloc(sizeof(float)*buff_size_src);
     float *dst_tst = (float *)malloc(sizeof(float)*buff_size_dst);



     long *results = (long *)malloc(sizeof(long)*num_trials);

     // run the test
     time_function_under_test(num_trials,
			      num_runs_per_trial,
			      results,
			      
			      m, n,
			      src_tst,
			      rs_src, cs_src,
			      dst_tst,
			      rs_dst, cs_dst);



     long min_res = pick_min_in_list(num_trials, results);

     // NOTE: since we are taking small measurements we do have to
     //       fudge a bit to overcome the overhead of timing.
     float nanoseconds = ((float)min_res)/(num_runs_per_trial);

     // size of matrix times 2 (reading and writing)
     long num_bytes = sizeof(float)*m*n*2;

     // This gives us throughput as GB/s
     float throughput =  num_bytes / nanoseconds;

     free(results);
     
     printf("%i, %i, %i, %i, %i, %i, %f\n",
	    m,n,
	    rs_src,cs_src,
	    rs_dst,cs_dst,
	    throughput );


     
     free(src_tst);
     free(dst_tst);

     
   }

  
}

