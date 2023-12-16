#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "cholesky.h"

#define Nq N*N

/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE POLYBENCH_1D(p, N, n),
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j;
  for (i = 0; i < n; i++)
  {
    p[i] = 1.0 / n;   
    for (j = 0; j < n; j++)
      A[i][j] = 1.0 / n;

  }
}

/* Linearization of the Matrix A */
static void matrix_linearization(int n, int nq,
                                 DATA_TYPE POLYBENCH_1D(A_lin, Nq, nq),
                                 DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A_lin[i*n + j] = A[i][j];
}                      

/* Check the correctness of the two output. 
    If difference in output is found between A and A_lin,
    it will be assert. */
static void check_correctness(int n, int nq,
                              DATA_TYPE POLYBENCH_1D(A_lin, Nq, nq),
                              DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      assert(A[i][j] == A_lin[i*N + j]);

  // If no assertion failures occurred, print a success message
  printf("Assertion passed: Each element in A is equal to the corresponding element in A_lin.\n");
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

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;
  int nq = N*N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);    // Matrix A.
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, N, n);          // Support struct.
  POLYBENCH_1D_ARRAY_DECL(A_lin, DATA_TYPE, Nq, nq);    // Matrix A linearization.

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));

  /* Linearize the matrix A in a 1D array [n*n]. */
  matrix_linearization(n, nq, POLYBENCH_ARRAY(A_lin), POLYBENCH_ARRAY(A));

  /* Check the correctness of the linearization. */
  check_correctness(n, nq, POLYBENCH_ARRAY(A_lin), POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky(n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));

  // TODO: Run gpu kernel.

  /* Check the correctness of the CPU and GPU/Device implementation. */
  check_correctness(n, nq, POLYBENCH_ARRAY(A_lin), POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // TODO: Add the DCE for linearization
  // polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));


  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(A_lin);
  POLYBENCH_FREE_ARRAY(p);

  return 0;
}
