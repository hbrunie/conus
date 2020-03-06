all: test_cpu.ex test_conus.ex test_gpu.ex test_gpu_nvtx.ex

CXXSTD:=-std=c++11
NVCXX=nvcc
CXX=g++

CXXFLAGS=$(CXXSTD) -g
INCLUDE=-I include -I.

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $(INCLUDE) -o $@ $<

%.o: %.cu
	$(NVCXX) $(CXXFLAGS) -c $(INCLUDE) -o $@ $<

test_cpu.ex: Random.o conus.o conus_cpu.o test_cpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^

test_conus.ex: conus.o simpleTest.o conus_gpu.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^

test_gpu.ex: Random.o conus.o conus_cpu.o test_gpu.o conus_gpu.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^

test_gpu_nvtx.ex: Random.o conus.o conus_cpu.o test_gpu_nvtx.o conus_gpu.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^ -lnvToolsExt

clean:
	rm -f *.o

cleanall: clean
	rm -f *.ex
