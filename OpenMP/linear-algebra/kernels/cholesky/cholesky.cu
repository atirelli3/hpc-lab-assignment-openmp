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

#define N 32

// #define DATA_TYPE float
#define Nq N*N

#define BLOCK_SIZE 16

/* Array initialization. */
static void init_array(int n, DATA_TYPE *p, DATA_TYPE *A)
{
  int i, j;
  for (i = 0; i < n; i++)
  {
    p[i] = 0;   
    for (j = 0; j < n; j++)
      A[i*n + j] = 1.0 / (i + j + 1);
  }

  for (i = 0; i < n; i++)
    A[i*n + i] += n;
}                    

bool areEqual(float a, float b, float epsilon = 1e-2) {
    return std::abs(a - b) < epsilon;
}

/* Check the correctness of the two output. 
    If difference in output is found between A and A_d,
    it will be assert. */
static void check_correctness(int n, DATA_TYPE *A_d, DATA_TYPE *A)
{
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (!(areEqual(A[i*n +j], A_d[i*n + j]) || std::isnan(A[i*n + j]))) {
        printf("Assertion failed: A[%d][%d] != A_d[%d][%d]. \n", i, j, i, j);
        fprintf(stderr, DATA_PRINTF_MODIFIER, A[i*n + j]);
        fprintf(stderr, DATA_PRINTF_MODIFIER, A_d[i*n + j]);
        return;
      }
    }
  }

  // If no assertion failures occurred, print a success message
  printf("Assertion passed: Each element in A is equal to the corresponding element in A_d.\n");
}

/* DCE code. Must scan the entire live-out data. */
static void print_dataset(int n, DATA_TYPE *Dataset)
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, Dataset[i*n + j]);
      if ((i * N + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
}

// /* DCE code. Must scan the entire live-out data. */
// static void print_dataset_linear(int n, int nq,
//                                  DATA_TYPE POLYBENCH_1D(A_lin, Nq, nq))
// {
//   int i, j;

//   for (i = 0; i < n; i++)
//     for (j = 0; j < n; j++)
//     {
//       fprintf(stderr, DATA_PRINTF_MODIFIER, A_lin[i*n + j]);
//       if ((i * N + j) % 20 == 0)
//         fprintf(stderr, "\n");
//     }
// }

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_cholesky(int n, DATA_TYPE *p, DATA_TYPE *A)
{
  int i, j, k;

  DATA_TYPE x;
  for (i = 0; i < _PB_N; ++i)
  {
    x = A[i*n + i];
    for (j = 0; j <= i - 1; ++j)
      x = x - A[i*n + j] * A[i*n + j];
    p[i] = 1.0 / sqrt(x);
    
    for (j = i + 1; j < _PB_N; ++j)
    {
      x = A[i*n + j];
      for (k = 0; k <= i - 1; ++k)
        x = x - A[j*n + k] * A[i*n + k];
      A[j*n + i] = x * p[i];
    }
  }
}

static void stream_cholesky(int n, DATA_TYPE *p, DATA_TYPE *A)
{
  int i;
  DATA_TYPE x;
  for (i = 0; i )
}

__global__ void device_cholesky(int n, DATA_TYPE *p, DATA_TYPE *A)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j, k;

  if (i < n) {
    DATA_TYPE x = A[i * n + i];

    for (j = 0; j <= i - 1; ++j)
      x = x - A[i * n + j] * A[i * n + j];
    p[i] = 1.0 / sqrt(x);

    // Ensure all threads have computed p[i] before using it
    __syncthreads();

    for (j = i + 1; j < n; ++j) {
      x = A[i * n + j];
      for (k = 0; k <= i - 1; ++k)
        x = x - A[j * n + k] * A[i * n + k];
      A[j * n + i] = x * p[i];
    }
  }
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;
  int nq = Nq;
  
  /* Variable declaration/allocation. */
  DATA_TYPE *p;     // Support struct.
  DATA_TYPE *A;     // Matrix A.
  DATA_TYPE *A_d;   // Matrix A linearization.

  /* Allocate pinned memory on the host. */
  cudaHostAlloc((void**)&p, N * sizeof(DATA_TYPE), cudaHostAllocDefault);
  cudaHostAlloc((void**)&A, N * N * sizeof(DATA_TYPE), cudaHostAllocDefault);
  cudaHostAlloc((void**)&A_d, Nq * sizeof(DATA_TYPE), cudaHostAllocDefault);

  /* Allocate device memory */
  DATA_TYPE *d_p, *d_A;
  cudaMalloc((void**)&d_p, N * sizeof(DATA_TYPE));
  cudaMalloc((void**)&d_A, Nq * sizeof(DATA_TYPE));

  /* Initialize array(s). */
  // init_array(n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));
  /* Initialize array(s). */
  init_array(n, p, A);
  cudaMemcpy(A_d, A, Nq * sizeof(DATA_TYPE), cudaMemcpyHostToHost);

  /* Linearize the matrix A in a 1D array [n*n]. */
  // matrix_linearization(n, nq, POLYBENCH_ARRAY(A_lin), POLYBENCH_ARRAY(A));

  /* Check the correctness of the linearization. */
  check_correctness(n, A, A_d);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky(n, p, A);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  cudaMemset(&p, 0, N * sizeof(DATA_TYPE));

  /* Start timer. */
  polybench_start_instruments;

  /* Copy data from pinned host memory to device memory. */
  cudaMemcpy(d_p, p, N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, A_d, Nq * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

  /* Run GPU kernel. */
  int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  device_cholesky<<<numBlocks, BLOCK_SIZE>>>(n, d_p, d_A);

  /* Copy results from device memory to pinned host memory. */
  cudaMemcpy(p, d_p, N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
  cudaMemcpy(A_d, d_A, Nq * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Check the correctness of the CPU and GPU/Device implementation. */
  check_correctness(n, A, A_d);

  /* Free device memory. */
  cudaFree(d_p);
  cudaFree(d_A);

  /* Be clean. */
  cudaFreeHost(p);
  cudaFreeHost(A);
  cudaFreeHost(A_d);

  return 0;
}
