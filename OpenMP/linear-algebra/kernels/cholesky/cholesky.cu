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

#define BLOCK_SIZE 32

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

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, Dataset[i*n + j]);
      if ((i * N + j) % 20 == 0)
        fprintf(stderr, "\n");
    }

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
  //int i, j, k;
//
  //DATA_TYPE x;
  //  for (i = 0; i < _PB_N; ++i) {
  //      p[i] = 1 / sqrt(A[i*n + i] - p[i]);
//
  //      #pragma omp parallel for private(j, k, x)
  //      for (j = i + 1; j < _PB_N; j++) {
  //          x = A[i*n + j];
  //          
  //          #pragma omp simd reduction(-:x)
  //          for (k = 0; k <= i - 1; ++k)
  //              x = x - A[j*n + k] * A[i*n + k];
//
  //          A[j*n + i] = x * p[i];
  //          p[j] += A[j*n + i] * A[j*n + i]; 
  //      }
  //  }
}
__global__ void device_cholesky_1(int n,
                                int i,
                                DATA_TYPE *p,
                                DATA_TYPE *A) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid == 0)
    p[i] = A[i * n + i];
  
  __syncthreads();

  DATA_TYPE tmp = 0;
  for (int j = 0; j < i; j += BLOCK_SIZE) {
    int index = j + tid;
    if (index < i) 
      tmp -= A[i * n + index] * A[i * n + index];
  }
  
  atomicAdd(&p[i], tmp);

  __syncthreads();

  if (tid == 0) 
    p[i] = 1 / sqrt(p[i]);
}

__global__ void device_cholesky_2(int n,
                                int i,
                                DATA_TYPE *p,
                                DATA_TYPE *A) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int j = tid + i + 1;

  __shared__ DATA_TYPE vec_shared[BLOCK_SIZE];

  DATA_TYPE tmp = (j < n) ? A[i*n + j] : 0;
  for (int bk = 0; bk < i; bk += BLOCK_SIZE) {
    int index = i * n + bk + tid;
    if (index < n)
      vec_shared[tid] = A[index];

    __syncthreads();

    for (int k = bk; j < n && k < bk + BLOCK_SIZE && k < i; k++)
      tmp -= vec_shared[k] * A[j*n + k];

    __syncthreads();
  }
  
  if (j < n)
    A[j*n + i] = p[i] * tmp;
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
  cudaHostAlloc((void**)&A_d, N * N * sizeof(DATA_TYPE), cudaHostAllocDefault);

  /* Allocate device memory */
  DATA_TYPE *d_p, *d_A;
  cudaMalloc((void**)&d_p, N * sizeof(DATA_TYPE));
  cudaMalloc((void**)&d_A, Nq * sizeof(DATA_TYPE));

  /* Initialize array(s). */
  init_array(n, p, A);

  cudaMemcpy(A_d, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToHost);


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


  print_dataset_matrix(n, A);
      fprintf(stderr, "\n----------------------\n");
      for (int k = 0; k < n; k++) {
        fprintf(stderr, "%.6f ", p[k]);
      }
      fprintf(stderr, "\n----------------------\n");

  cudaMemset(&p, 0, N * sizeof(DATA_TYPE));


  /* Run GPU kernel. */

  polybench_start_instruments;
  /* Copy data from pinned host memory to device memory. */
  cudaMemcpy(d_p, p, N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, A_d, Nq * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
  
  for (int i = 0; i < N; i++) {
    device_cholesky_1<<<1, BLOCK_SIZE>>>(n, i, d_p, d_A);

    if (i < n - 1) {
      int numBlocks = (N - i - 2 + BLOCK_SIZE) / BLOCK_SIZE;
      device_cholesky_2<<<numBlocks, BLOCK_SIZE>>>(n, i, d_p, d_A);
    }
  }
  /* Copy results from device memory to pinned host memory. */
  cudaMemcpy(p, d_p, N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
  cudaMemcpy(A_d, d_A, Nq * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

  polybench_stop_instruments;
  polybench_print_instruments;

  print_dataset_matrix(n, A_d);
      fprintf(stderr, "\n----------------------\n");
      for (int k = 0; k < n; k++) {
        fprintf(stderr, "%.6f ", p[k]);
      }
      fprintf(stderr, "\n----------------------\n");

  /* Check the correctness of the CPU and GPU/Device implementation. */
  check_correctness(n, nq, A_d, A);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // polybench_prevent_dce(print_dataset_matrix(n, POLYBENCH_ARRAY(A)));
  // polybench_prevent_dce(print_dataset_linear(n, nq, POLYBENCH_ARRAY(A_d)));

  /* Free device memory. */
  cudaFree(d_p);
  cudaFree(d_A);

  /* Be clean. */
  cudaFreeHost(p);
  cudaFreeHost(A);
  cudaFreeHost(A_d);

  return 0;
}
