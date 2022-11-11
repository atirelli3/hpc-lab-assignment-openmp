# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O2

all: bfs

bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs 

clean:
	rm -f bfs bfs_offload result.txt

run: bfs
	./run
