
CXXSTD:=-std=c++11

%.o: %.cpp
	$(CXX) $(CXXSTD) -c -I. -o $@ $<

test_random.ex: Random.o test_random.o
	$(CXX) $(CXXSTD) -o $@ $^


clean:
	rm *.o
