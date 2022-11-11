##Available Benchmarks

####datamining
* correlation
* covariance

####linear-algebra/kernels
* 2mm
* 3mm
* atax
* bicg
* cholesky
* doitgen
* gemm
* gemver
* gesummv
* mvt
* symm
* syr2k
* syrk
* trisolv
* trmm

####linear-algebra/solvers
* durbin
* dynprog
* gramschmidt
* lu
* ludcmp

####stencils
* adi
* convolution-2d
* convolution-3d
* fdtd-2d
* jacobi-1d-imper
* jacobi-2d-imper
* seidel-2d

Modifying Codes
------------------

Parameters such as the input sizes, data type, and threshold for GPU-CPU output comparison can be modified using constants
within the codes and .h files.  After modifying, run `make clean` then `make` on relevant code for modifications to take effect in resulting executable.

###Parameter Configuration:

####Input Size:
By default the `STANDARD_DATASET` as defined in the `.cuh/.h` file is used as the input size.  The dataset choice can be adjusted from `STANDARD_DATASET` to other
options (`MINI_DATASET`, `SMALL_DATASET`, etc) in the `.cuh/.h` file, the dataset size can be adjusted by defining the input size manually in the `.cuh/.h` file, or
the input size can be changed by simply adjusting the `STANDARD_DATASET` so the program has different input dimensions.

###`DATA_TYPE` (in `.cuh/.h` files):
By default, the `DATA_TYPE` used in these codes are `float` that can be changed to `double` by changing the `DATA_TYPE` typedef. Note that in OpenCL, the `DATA_TYPE` needs to be changed in both the .h and .cl files, as the .cl files contain the kernel code and is compiled separately at run-time.


####Other available options

These are passed as macro definitions during compilation time 
(e.g `-Dname_of_the_option`) or can be added with a `#define` to the code.
- `POLYBENCH_STACK_ARRAYS` (only applies to allocation on host): 
use stack allocation instead of malloc [default: off]
- `POLYBENCH_DUMP_ARRAYS`: dump all live-out arrays on stderr [default: off]
- `POLYBENCH_CYCLE_ACCURATE_TIMER`: Use Time Stamp Counter to monitor
  the execution time of the kernel [default: off]
- `MINI_DATASET`, `SMALL_DATASET`, `STANDARD_DATASET`, `LARGE_DATASET`,
  `EXTRALARGE_DATASET`: set the dataset size to be used
  [default: `STANDARD_DATASET`]

- `POLYBENCH_USE_C99_PROTO`: Use standard C99 prototype for the functions.

- `POLYBENCH_USE_SCALAR_LB`: Use scalar loop bounds instead of parametric ones.
