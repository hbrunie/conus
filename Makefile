all: test_random.ex test_conus.ex

CXXSTD:=-std=c++11
NVCXX=nvcc
CXX=g++

INCLUDE=-I include -I.

%.o: %.cpp
	$(CXX) $(CXXSTD) -c $(INCLUDE) -o $@ $<

%.o: %.cu
	$(NVCXX) $(CXXSTD) -c $(INCLUDE) -o $@ $<

test_random.ex: Random.o test_random.o
	$(CXX) $(CXXSTD) -o $@ $^

test_conus.ex: simpleTest.o conus_gpu.o
	$(NVCXX) $(CXXSTD) -o $@ $^

clean:
	rm -f *.o

cleanall: clean
	rm -f *.ex
