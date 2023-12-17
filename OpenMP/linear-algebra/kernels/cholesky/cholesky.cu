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

/* Array initialization. */
static void init_array(int n, DATA_TYPE *p, DATA_TYPE *A, DATA_TYPE *B)
{
  int i, j;
  for (i = 0; i < n; i++)
  {
    p[i] = 0;   
    for (j = 0; j < n; j++)
    {
      A[i*n + j] = 1.0 / (i + j + 1);
      B[i*n + j] = A[i*n + j];
    }
      
  }

  for (i = 0; i < n; i++)
  {
    A[i*n + i] += n;
    B[i*n + i] = A[i*n + i];
  }
    
}
              
bool areEqual(float a, float b, float epsilon = 1e-3) {
    return std::abs(a - b) < epsilon;
}

/* Check the correctness of the two output. 
  If difference in output is found between A and A_d,
  it will be assert. */
static void check_correctness(int n, DATA_TYPE *A, DATA_TYPE *B)
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

static void print_dataset(int n, DATA_TYPE *dataset)
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

static void device_cholesky(int n, DATA_TYPE *p, DATA_TYPE *A)
{
  int i;
  DATA_TYPE x;
  for (i = 0; i < _PB_N; i++) {
    x = A[i*n + i];
  }
}

// __global__ void device_cholesky_1(int n, int i, DATA_TYPE *p, DATA_TYPE *A) 
// {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
//   if (tid == 0)
//     p[i] = A[i * n + i];
  
//   __syncthreads();

//   DATA_TYPE tmp = 0;
//   for (int j = 0; j < i; j += BLOCK_SIZE) {
//     int index = j + tid;
//     if (index < i) 
//       tmp -= A[i * n + index] * A[i * n + index];
//   }
  
//   atomicAdd(&p[i], tmp);

//   __syncthreads();

//   if (tid == 0) 
//     p[i] = 1 / sqrt(p[i]);
// }

// __global__ void device_cholesky_2(int n, int i, DATA_TYPE *p, DATA_TYPE *A) 
// {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int j = tid + i + 1;

//   __shared__ DATA_TYPE vec_shared[BLOCK_SIZE];

//   DATA_TYPE tmp = (j < n) ? A[i*n + j] : 0;
//   for (int bk = 0; bk < i; bk += BLOCK_SIZE) {
//     int index = i * n + bk + tid;
//     if (index < n)
//       vec_shared[tid] = A[index];

//     __syncthreads();

//     for (int k = bk; j < n && k < bk + BLOCK_SIZE && k < i; k++)
//       tmp -= vec_shared[k] * A[j*n + k];

//     __syncthreads();
//   }
  
//   if (j < n)
//     A[j*n + i] = p[i] * tmp;
// }

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE *p, *A, *A_d;

  /* Allocate in UVM */
  cudaMallocManaged((void **)&p, sizeof(DATA_TYPE) * n);      // Allocate p.
  cudaMallocManaged((void **)&A, sizeof(DATA_TYPE) * n * n);  // Allocate matrix (lin) A.
  cudaMallocManaged((void **)&A_d, sizeof(DATA_TYPE) * n * n);  // Allocate matrix (lin) A.

  /* Initialize array(s). */
  init_array(n, p, A, A_d);

  /* Check initialization math consistency. */
  check_correctness(n, A, A_d);
}