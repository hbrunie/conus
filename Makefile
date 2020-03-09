CXX      := g++
NVCXX    := nvcc

CXXSTD   := -std=c++11
CXXFLAGS := $(CXXSTD) -g -Xcompiler -fopenmp
INCLUDE  := -I include -I extern/Random123-1.13.2/tests
INCLUDE  += -I extern/Random123-1.13.2/include
INCLUDE  += -I extern/Random123-1.13.2/examples

SRC_DIR  := src
APP_DIR  := tests

all: test_cpu.ex test_gpu.ex test_gpu_nvtx.ex profile_gpu.ex

%.o: $(SRC_DIR)/%.cpp
	$(NVCXX) $(CXXFLAGS) -c $(INCLUDE) -o $@ $<

%.o: $(SRC_DIR)/%.cu
	$(NVCXX) $(CXXFLAGS) -c $(INCLUDE) -o $@ $<

%.o: $(APP_DIR)/%.cpp
	$(NVCXX) $(CXXFLAGS) -c $(INCLUDE) -o $@ $<

%.o: $(APP_DIR)/%.cu
	$(NVCXX) $(CXXFLAGS) -c $(INCLUDE) -o $@ $<

test_cpu.ex: Random.o conus_cpu.o test_cpu.o conus.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^

test_gpu.ex: Random.o conus_cpu.o test_gpu.o conus_gpu.o conus.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^

profile_gpu.ex: Random.o conus_cpu.o profile_gpu.o conus_gpu.o conus.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^

test_gpu_nvtx.ex: Random.o conus_cpu.o test_gpu_nvtx.o conus_gpu.o conus.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^ -lnvToolsExt

clean:
	rm -f *.o

cleanall: clean
	rm -f *.ex *.o
