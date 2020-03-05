#include <iostream>
#include "example_seeds.h"



// need to be defined to use Random123
int debug = 0;
const char *progname;


void conusInit(){
    progname = "Prog using Conus";
}

void conusFinalize(){
}



unsigned long getULseed(){
    unsigned long seed = 0xdeadbeef12345678;
    return 0;
}



void deleteRandomsCPU(double * arr){
    free(arr);
}
