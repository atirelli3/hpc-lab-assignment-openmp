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

#define BLOCK_SIZE 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  bool test = code == cudaSuccess;
  // cout << "code " << std::boolalpha<< test;
   if (code != cudaSuccess)
   {
      // const char *errorStr = NULL;
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE *p,
                       DATA_TYPE *A)
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
              
bool areEqual(float a, float b, float epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

/* Check the correctness of the two output. 
    If difference in output is found between A and A_d,
    it will be assert. */
static void check_correctness(int n, int nq,
                              DATA_TYPE *A_d,
                              DATA_TYPE *A)
{
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (!(areEqual(A[i*n +j], A_d[i*n + j]) || std::isnan(A[i*n + j]))) {
        printf("Assertion failed: A[%d][%d] != A_d[%d][%d]. \n", i, j, i, j);
        return;
      }
    }
  }

  // If no assertion failures occurred, print a success message
  printf("Assertion passed: Each element in A is equal to the corresponding element in A_d.\n");
}

/* DCE code. Must scan the entire live-out data. */
static void print_dataset_matrix(int n,
                                DATA_TYPE *A)

{
  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i*n + j]);
    }

        fprintf(stderr, "\n");
  }
}

/* DCE code. Must scan the entire live-out data. */
static void print_dataset_linear(int n, int nq,
                                 DATA_TYPE *A_d)
{
  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A_d[i*n + j]);
    }

    fprintf(stderr, "\n");
  }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_cholesky(int n,
                            DATA_TYPE *p,
                            DATA_TYPE *A)
{
  int i, j, k;

  DATA_TYPE x;
    for (i = 0; i < _PB_N; ++i) {
        p[i] = 1 / sqrt(A[i*n + i] - p[i]);

        #pragma omp parallel for private(j, k, x)
        for (j = i + 1; j < _PB_N; j++) {
            x = A[i*n + j];
            
            #pragma omp simd reduction(-:x)
            for (k = 0; k <= i - 1; ++k)
                x = x - A[j*n + k] * A[i*n + k];

            A[j*n + i] = x * p[i];
            p[j] += A[j*n + i] * A[j*n + i]; 
        }
    }
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
  p[j] += A[j*n + i] * A[j*n + i];
}

__global__ void device_cholesky(int n,
                                int i,
                                DATA_TYPE *p,
                                DATA_TYPE *A) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int j = tid + i + 1;

  __shared__ DATA_TYPE vec_shared[BLOCK_SIZE];

  DATA_TYPE tmp = (j < n) ? A[i*n + j] : 0;
  for (int bk = 0; bk < i; bk += BLOCK_SIZE) {
    int index = bk + tid;
    if (index < n)
      vec_shared[tid] = A[i * n + index];

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
  int nq = N*N;

  /* Variable declaration/allocation. */
  DATA_TYPE *p, *p_d, *A, *A_d;

  p = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  A = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));

  /* Allocate pinned memory on the host. */
  gpuErrchk(cudaMallocManaged((void**)&p_d, N * sizeof(DATA_TYPE)));
  gpuErrchk(cudaMallocManaged((void**)&A_d, N * N * sizeof(DATA_TYPE)));

  /* Allocate device memory */

  /* Initialize array(s). */
  init_array(n, p, A);
  init_array(n, p_d, A_d);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky(n, p, A);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;


  //print_dataset_matrix(n, A);
  //    fprintf(stderr, "\n----------------------\n");
  //    for (int k = 0; k < n; k++) {
  //      fprintf(stderr, "%.6f ", p[k]);
  //    }
  //    fprintf(stderr, "\n----------------------\n");


  /* Run GPU kernel. */

  polybench_start_instruments;
  /* Copy data from pinned host memory to device memory. */
  
  for (int i = 0; i < N; i++) {
    DATA_TYPE a = p_d[i];
    DATA_TYPE b = A_d[i*n + i];
    DATA_TYPE tmp = 1 / sqrt(b - a);
    p_d[i] = tmp;

    if (i < n - 1) {
      int numBlocks = (N - i - 2 + BLOCK_SIZE) / BLOCK_SIZE;
      // device_cholesky<<<numBlocks, BLOCK_SIZE>>>(n, i, p_d, A_d);
      compute_A<<<numBlocks, BLOCK_SIZE>>>(n, i, p_d, A_d);
      cudaDeviceSynchronize();
    }
  }

  gpuErrchk(cudaPeekAtLastError());

  /* Copy results from device memory to pinned host memory. */

  polybench_stop_instruments;
  polybench_print_instruments;

  //print_dataset_matrix(n, A_d);
  //    fprintf(stderr, "\n----------------------\n");
  //    for (int k = 0; k < n; k++) {
  //      fprintf(stderr, "%.6f ", p_d[k]);
  //    }
  //    fprintf(stderr, "\n----------------------\n");

  /* Check the correctness of the CPU and GPU/Device implementation. */
  check_correctness(n, nq, A_d, A);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // polybench_prevent_dce(print_dataset_matrix(n, POLYBENCH_ARRAY(A)));
  // polybench_prevent_dce(print_dataset_linear(n, nq, POLYBENCH_ARRAY(A_d)));


  /* Be clean. */
  cudaFree(p_d);
  cudaFree(A_d);

  
  free(p);
  free(A);

  return 0;
}
