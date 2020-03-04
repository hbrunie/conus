#include <conus.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
using namespace std;

// return the number of differences between 2 arrays
// of double, diff if > tolerance.
template <typename T>
unsigned checkRandomNumbersEquivalence(
        T* array_A,T* array_B, double tol, unsigned arrSize){
    unsigned nbDiff=0;
    unsigned cnt = 0;
    for(unsigned i=0; i<arrSize; i++){
        cerr << setprecision(15);
        cerr << array_A[i]<<endl;
    //   cerr <<array_B[i] <<endl;
        if(abs(array_A[i] - array_B[i]) > tol){
            nbDiff ++;
            if(cnt<10){
                cnt++;
                cerr << "index: " << i <<
                    " array_A: " << array_A[i] <<
                    " array_B: " << array_B[i] <<endl;
            }
        }
    }
    return nbDiff;
}

int main(int argc, char ** argv) {
    unsigned long N = 1024;
    double tol = 0.001;
    unsigned nbDiffs = 0;
    std::cout << "default array size is 1024; You can change: "
        << argv[0] << " N" << std::endl;
    if(argc>1)
        N = (unsigned long) atol(argv[1]);
    std::cout << "default tolerance is 0.001; You can change: "
        << argv[0] << " N tol" << std::endl;
    if(argc>2)
        tol = (double) atof(argv[2]);
    conusInit();
    double* randCPU = generateRandomsCPU<double>(N);
    double* randGPU = generateRandomsGPUd(N);
    nbDiffs = checkRandomNumbersEquivalence<double>(randCPU,randGPU,tol, N);
    //deleteRandomsCPU(randCPU);
    //deleteRandomsCPU(randGPU);
    //conusFinalize();

    std::cout << "nbDiffs" << nbDiffs
              << std::endl;

    return 0;
}
