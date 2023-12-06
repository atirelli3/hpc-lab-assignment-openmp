#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "cholesky.h"

/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE POLYBENCH_1D(p, N, n),
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                       DATA_TYPE POLYBENCH_1D(A_opt, N*N, n*n))
{
  int i, j;

  for (i = 0; i < n; i++)
  {
    p[i] = 1.0 / n;
    for (j = 0; j < n; j++)
      A[i][j] = 1.0 / n;
      A_opt[i*n + j] = 1.0 / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_2D(A, N, N, n, n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * N + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_cholesky(int n,
                            DATA_TYPE POLYBENCH_1D(p, N, n),
                            DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j, k;

  DATA_TYPE x;
  for (i = 0; i < _PB_N; ++i)
  {
    x = A[i][i];
    for (j = 0; j <= i - 1; ++j)
      x = x - A[i][j] * A[i][j];
    p[i] = 1.0 / sqrt(x);
    for (j = i + 1; j < _PB_N; ++j)
    {
      x = A[i][j];
      for (k = 0; k <= i - 1; ++k)
        x = x - A[j][k] * A[i][k];
      A[j][i] = x * p[i];
    }
  }
}


/* 
* Main computational kernel on the device (GPU). 
* 
* The whole function will be timed, including the call and return. 
*/
static void device_cholesky(int n,
                            DATA_TYPE POLYBENCH_1D(p, N, n),
                            DATA_TYPE POLYBENCH_1D(A_opt, N, n))
{

}

// TODO: Fix the output correctness.
/* Main computational kernel optimize. The whole function will be timed,
   including the call and return. */
static void opt_kernel_cholesky(int n,
                            DATA_TYPE POLYBENCH_1D(p, N, n),
                            DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j, k;

  DATA_TYPE x;
    for (i = 0; i < _PB_N; ++i) {
        p[i] = 1 / sqrt(A[i][i] - p[i]);

        #pragma omp parallel for private(j, k, x)
        for (j = i + 1; j < _PB_N; j++) {
            x = A[i][j];
            
            #pragma omp simd reduction(-:x)
            for (k = 0; k <= i - 1; ++k)
                x = x - A[j][k] * A[i][k];

            A[j][i] = x * p[i];
            p[j] += A[j][i] * A[j][i]; 
        }
    }
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  // TODO: Linearize the array (1D) A with dimension N*N
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(A_opt, DATA_TYPE, N*N, n*n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_opt));

  /* Start timer. */
  polybench_start_instruments;

  // TODO: Run cpu kernel.

  // TODO: Run gpu kernel.

  // TODO: Assert the CPU and GPU for correctness of the output..

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // TODO: Add the DCE for A_opt
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));


  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYYBENCH_FREE_ARRAY(A_opt);
  POLYBENCH_FREE_ARRAY(p);

  return 0;
}
