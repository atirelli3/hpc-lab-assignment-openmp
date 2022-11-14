all: euler3d_cpu

euler3d_cpu: euler3d_cpu.cpp
	g++ -O3 -Wall -fopenmp euler3d_cpu.cpp -o euler3d_cpu

run: euler3d_cpu
	./run

clean:
	rm -rf euler3d_cpu
	rm -rf density density_energy momentum
