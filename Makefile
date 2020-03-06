all: test_cpu.ex test_gpu.ex test_gpu_nvtx.ex profile_gpu.ex

CXXSTD:=-std=c++11
NVCXX=nvcc
CXX=g++

CXXFLAGS=$(CXXSTD) -g -Xcompiler -fopenmp
INCLUDE=-I include -I.

%.o: %.cpp
	$(NVCXX) $(CXXFLAGS) -c $(INCLUDE) -o $@ $<

%.o: %.cu
	$(NVCXX) $(CXXFLAGS) -c $(INCLUDE) -o $@ $<

test_cpu.ex: Random.o conus.o conus_cpu.o test_cpu.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^

test_gpu.ex: Random.o conus.o conus_cpu.o test_gpu.o conus_gpu.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^

profile_gpu.ex: Random.o conus.o conus_cpu.o profile_gpu.o conus_gpu.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^

test_gpu_nvtx.ex: Random.o conus.o conus_cpu.o test_gpu_nvtx.o conus_gpu.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^ -lnvToolsExt

clean:
	rm -f *.o

cleanall: clean
	rm -f *.ex
