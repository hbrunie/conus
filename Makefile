all: test_random.ex test_conus.ex

CXXSTD:=-std=c++11
NVCXX=nvcc
CXX=g++

CXXFLAGS=$(CXXSTD) -g
INCLUDE=-I include -I.

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $(INCLUDE) -o $@ $<

%.o: %.cu
	$(NVCXX) $(CXXFLAGS) -c $(INCLUDE) -o $@ $<

test_random.ex: Random.o conus_random.o test_random.o
	$(CXX) $(CXXFLAGS) -o $@ $^

test_conus.ex: simpleTest.o conus_gpu.o
	$(NVCXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f *.o

cleanall: clean
	rm -f *.ex
