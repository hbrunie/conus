all: test
CXX=g++
INCLUDE=-I include -g -I.

test: simpleTest.cpp conus_gpu.o
	nvcc -o $@ $^ $(INCLUDE)

conus_cpu.o: conus_cpu.cpp
	nvcc -o $@ -c $^ $(INCLUDE)

conus_gpu.o: conus_gpu.cu
	nvcc -o $@ -c $^ $(INCLUDE) -arch=compute_70

clean:
	rm -f *.o

cleanall: clean
	rm -f test
