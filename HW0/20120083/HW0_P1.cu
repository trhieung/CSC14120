#include<stdio.h>

void printDeviceInfo()
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("*****************************Cau 1*****************************\n");
    printf("GPU card's name:\t\t\t\t%s\n", devProp.name);
    printf("GPU computation capabilities:\t\t\t %d.%d\n", devProp.major, devProp.minor);
    printf("Maximum number of block dimensions:\t\t %d\n", devProp.maxThreadsPerBlock);
    printf("Maximum number of grid dimensions:\t\t %d, %d, %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
    printf("Maximum size of GPU memory:\t\t\t %zu bytes\n", devProp.totalGlobalMem);
    printf("Amount of constant memory:\t\t\t %zu bytes\n", devProp.totalConstMem);
    printf("Amount of shared memory per block:\t\t %zu bytes\n", devProp.sharedMemPerBlock);
    printf("Warp size:\t\t\t\t\t %d threads\n", devProp.warpSize);
}

int main(int argc, char ** argv){
    
    printDeviceInfo();
    return 0;
}