all: euler3d_cpu

euler3d_cpu: euler3d_cpu.cpp
	g++ -O3 -Wall -fopenmp euler3d_cpu.cpp -o euler3d_cpu

euler3d_omp: euler3d_solution.cpp
	g++ -O3 -Wall -fopenmp euler3d_solution.cpp -o euler3d_omp

run: euler3d_cpu
	./run

run_omp: euler3d_omp
	./run-omp
	
clean:
	rm -rf euler3d_cpu euler3d_omp
	rm -rf density density_energy momentum
