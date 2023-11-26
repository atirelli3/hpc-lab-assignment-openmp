#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "cholesky.h"

/*
 * The MatrixVector structure represents a mathematical matrix and a vector.
 *
 * It contains two members:
 * 1. p: This is an array of DATA_TYPE which represents a vector of size N.
 * 2. A: This is an array of DATA_TYPE which represents a matrix of size N*N.
 *
 * This structure can be used to perform various matrix and vector operations.
 */
typedef struct
{ 
  DATA_TYPE p[N];  // Vector of size N
  DATA_TYPE A[N * N];  // Matrix of size N*N
} MatrixVector;


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

/*
 * The mem_init_array function initializes the members of a MatrixVector structure.
 *
 * Parameters:
 * - n: The size of the matrix and vector.
 * - mv: A pointer to the MatrixVector structure to be initialized.
 *
 * This function initializes the vector 'p' and the matrix 'A' of the MatrixVector structure 
 * with the value 1.0/n. This means that after the function call, all elements of 'p' and 'A' 
 * will be equal to 1.0/n.
 */
static void mem_init_array(int n, MatrixVector* mv)
{
  int i, j;

  for (i = 0; i < n; i++)
  {
    mv->p[i] = 1.0 / n;
    for (j = 0; j < n; j++)
      mv->A[i * n + j] = 1.0 / n;
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

/*
 * The mem_print_array function prints the matrix 'A' of a MatrixVector structure.
 *
 * Parameters:
 * - n: The size of the matrix.
 * - mv: A pointer to the MatrixVector structure whose matrix is to be printed.
 *
 * This function iterates over each element of the matrix 'A' and prints it to the stderr.
 * After every 20 elements, it prints a newline character. This is done to ensure that the 
 * output is neatly formatted and easy to read.
 */
static void mem_print_array(int n, MatrixVector* mv)
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, mv->A[i * n + j]);
      if ((i * n + j) % 20 == 0)
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
 * The mem_kernel_cholesky function performs the Cholesky decomposition on a MatrixVector structure.
 *
 * Parameters:
 * - n: The size of the matrix.
 * - mv: A pointer to the MatrixVector structure to be decomposed.
 *
 * This function performs the Cholesky decomposition on the matrix 'A' of the MatrixVector structure.
 * The Cholesky decomposition is a decomposition of a Hermitian, positive-definite matrix into the 
 * product of a lower triangular matrix and its conjugate transpose. The result of the decomposition 
 * is stored back into the matrix 'A' and the vector 'p' of the MatrixVector structure.
 *
 * This function uses OpenMP directives for parallelization. The 'reduction' clause is used to 
 * perform a reduction on the variable 'x' in a thread-safe manner. The 'private' clause is used 
 * to specify that the variables 'x' and 'k' are private to each thread.
 */
static void mem_kernel_cholesky(int n, MatrixVector* mv)
{
  int i, j, k;

  DATA_TYPE x;
  for (i = 0; i < n; ++i)
  {
    x = mv->A[i * n + i];
    #pragma omp parallel for reduction(-:x)
    for (j = 0; j <= i - 1; ++j)
      x = x - mv->A[i * n + j] * mv->A[i * n + j];
    mv->p[i] = 1.0 / sqrt(x);
    #pragma omp parallel for private(x, k)
    for (j = i + 1; j < n; ++j)
    {
      x = mv->A[i * n + j];
      #pragma omp simd reduction(-:x)
      for (k = 0; k <= i - 1; ++k)
        x = x - mv->A[j * n + k] * mv->A[i * n + k];
      mv->A[j * n + i] = x * mv->p[i];
    }
  }
}


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
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, N, n);

  #ifdef MEM_OPT
    /* Initialize the linearization struct. */
    mem_init_array(n, mv);
  #else
    /* Initialize array(s). */
    init_array(n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));
  #endif

  /* Start timer. */
  polybench_start_instruments;

  #ifdef MEM_OPT
    /* Run memory optimize kernel. */
    mem_kernel_cholesky(n, mv);
  #else
  {
    #ifdef PARALLEL_OPT
      /* Run optimize kernel. */
      opt_kernel_cholesky(n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));
    #else
      /* Run kernel. */
      kernel_cholesky(n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));
    #endif
  }
  #endif

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(p);

  return 0;
}
