
CXXSTD:=-std=c++11

% : %.cpp
	$(CXX) $(CXXSTD) -o $@


all: Random.o

clean:
	rm *.o
