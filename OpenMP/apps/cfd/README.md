# CFD Solver
The CFD solver is an unstructured grid finite volume solver for the three-dimensional Euler equations for compressible flow. Effective GPU memory bandwidth is improved by reducing total global memory access and overlapping redundant computation, as well as using an appropriate numbering scheme and data layout.

We'd like to acknowledge Andrew Corrigan, Fernando Camelli, Rainald Lohner and John Wallin from George Mason University to contribute their codes to Rodinia.

For more details about this application, please visit http://web.cos.gmu.edu/~acorriga/pubs/gpu_cfd/

Paper: Andrew Corrigan, Fernando Camelli, Rainald Lohner and John Wallin. Running Unstructured Grid CFD Solvers on Modern Graphics Hardware. In Proceedings of the 19th AIAA Computational Fluid Dynamics Conference, June 2009.
