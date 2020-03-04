typedef long int int64_t;
// Generate N random numbers (double or single)
// Either on GPU or CPU using random123 library
// randomNumbers array is allocated on HOST
// directly by the method
template<typename T>
T* generateRandomsCPU(unsigned long N);

// randomNumbers array must be allocated on device
// Before calling the method
// TODO: do device allocation inside method

double*
generateRandomsGPUd(unsigned long N);

int64_t*
generateRandomsGPUi(unsigned long N);

#include "conus_cpu.cpp"

//void
//conusInit();
//
//void
//conusFinalize();
//void
//deleteRandomsCPU(double *);
//void
//deleteRandomsGPU(double *);
