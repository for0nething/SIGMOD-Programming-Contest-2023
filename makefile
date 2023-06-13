CC = g++

OPTION = -I./ -DIN_PARALLEL  -I /usr/local/include/eigen3 -fopenmp -Wall -march=native -ffast-math -flto -I$(MKLROOT)/include -DNDEBUG
LFLAGS = -std=c++11 -O3 $(OPTION)  -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core  -lmkl_gnu_thread -lpthread -lm -ldl


all: knng

knng : knn-construction-kgraph.cc
	$(CC) $(LFLAGS) knn-construction-kgraph.cc -o $@ -lkgraph -lboost_timer -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl


clean :
	rm knng
