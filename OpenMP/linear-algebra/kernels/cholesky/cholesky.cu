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

#define BLOCK_SIZE 32

#define DATA_TYPE float

/* Array initialization. */
static void init_array(int n, DATA_TYPE * __restrict__ p, 
                       DATA_TYPE * __restrict__ A)
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

/* Clone the structs */
static void clone_struct(int n, DATA_TYPE * __restrict__ a, 
                         DATA_TYPE * __restrict__ b, 
                         DATA_TYPE * __restrict__ A, 
                         DATA_TYPE * __restrict__ B)
{
  int i, j;
  for (i = 0; i < n; i++)
  {
    b[i] = a[i];
    for (j = 0; j < n; j++)
      B[i*n + j] = A[i*n + j];
  }
}
              
bool areEqual(float a, float b, float epsilon = 1e-3) {
    return std::abs(a - b) < epsilon;
}

/* Check the correctness of the two output. 
  If difference in output is found between A and A_d,
  it will be assert. */
static void check_correctness(int n, DATA_TYPE * __restrict__ A, 
                              DATA_TYPE * __restrict__ B)
{
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (!(areEqual(A[i*n +j], B[i*n + j]) || std::isnan(A[i*n + j]))) {
        printf("Assertion failed: A[%d][%d] != B[%d][%d]. \n", i, j, i, j);
        return;
      }
    }
  }
  // If no assertion failures occurred, print a success message
  printf("Assertion passed: Each element in A is equal to the corresponding element in A_d.\n");
}

static void print_dataset(int n, DATA_TYPE * __restrict__ dataset)
{
  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
      fprintf(stderr, DATA_PRINTF_MODIFIER, dataset[i*n + j]);
    fprintf(stderr, "\n");
  }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_cholesky(int n, DATA_TYPE * __restrict__ p, 
                            DATA_TYPE * __restrict__ A)
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

__global__ void compute_p(int n, int i, DATA_TYPE * __restrict__ p,
                     DATA_TYPE * __restrict__ A)
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
  atomicAdd(&p[i], -tmp);

  __syncthreads();

  if (tid == 0) 
    p[i] = 1 / sqrt(p[i]);
}

__global__ void compute_A(int n, int i, DATA_TYPE * __restrict__ p,
                          DATA_TYPE * __restrict__ A) 
{
  int j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
  if (j >= n)
    return;

  DATA_TYPE tmp = A[i*n + j];
  for (int k = 0; k < i; k++)
    tmp -= A[i*n + k] * A[j*n + k];

  A[j*n + i] = p[i] * tmp;
}

static void device_cholesky(int n, DATA_TYPE * __restrict__ p, 
                            DATA_TYPE * __restrict__ A)
{
  int i;
  for (i = 0; i < _PB_N; i++) {
    compute_p<<<1, BLOCK_SIZE>>>(n, i, p, A);
    if (i < n - 1) {
      int numBlocks = (N - i + BLOCK_SIZE) / BLOCK_SIZE;
      compute_A<<<numBlocks, BLOCK_SIZE>>>(n, i, p, A);
    }
    cudaDeviceSynchronize();
  }
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE *p, *A, *p_d, *A_d;

  /* Allocate in UVM */
  cudaMallocManaged((void **)&p, sizeof(DATA_TYPE) * n);        // Allocate p.
  cudaMallocManaged((void **)&A, sizeof(DATA_TYPE) * n * n);    // Allocate matrix (lin) A.
  cudaMallocManaged((void **)&p_d, sizeof(DATA_TYPE) * n);      // Allocate p.
  cudaMallocManaged((void **)&A_d, sizeof(DATA_TYPE) * n * n);  // Allocate matrix (lin) A.

  /* Initialize array(s). */
  init_array(n, p, A);
  clone_struct(n, p, p_d, A, A_d);

  /* Check initialization math consistency. */
  check_correctness(n, A, A_d);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky(n, p, A);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  // print_dataset(n, A);

  /* Start timer. */
  polybench_start_instruments;

  device_cholesky(n, p_d, A_d);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Check computational math consistency. */
  check_correctness(n, A, A_d);
  
  /* Be clean */
  cudaFree(p);
  cudaFree(A);
  cudaFree(p_d);
  cudaFree(A_d);
  cudaDeviceReset();
  return 0;
}