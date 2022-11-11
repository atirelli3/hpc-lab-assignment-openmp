#makefile

all: particle_filter.c
	gcc -O3 -fopenmp particle_filter.c -o particle_filter -lm 

clean:
	rm -rf particle_filter

run: all
	./run