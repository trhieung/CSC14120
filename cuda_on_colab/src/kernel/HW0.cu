#include "./HW0.h"

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

void addVecOnHost(float* in1, float* in2, float* out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = in1[i] + in2[i];    
}


// Each thread block processes 2 * blockDim.x consecutive elements that form two sections
__global__ void addVecOnDeviceVersion1(float* in1, float* in2, float* out, int n)
{    
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (i < n){
        out[i] = in1[i] + in2[i];
        out[i+blockDim.x] = in1[i+blockDim.x] + in2[i+blockDim.x];
    }
}

// use each thread to calculate two adjacent elements of a vector addition.
__global__ void addVecOnDeviceVersion2(float* in1, float* in2, float* out, int n)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i < n){
        out[i] = in1[i] + in2[i];
        out[i+1] = in1[i+1] + in2[i+1];
    }
}

void addVec(float* in1, float* in2, float* out, int n, int ver=0)
{
    GpuTimer timer;

    if (ver == 0){
		timer.Start();
		addVecOnHost(in1, in2, out, n);
		timer.Stop();
        
        float time = timer.Elapsed();
        printf("%18.9lf", time);    
    }
    else {
        // cudaDeviceProp devProp;
        // cudaGetDeviceProperties(&devProp, 0);
        // printf("GPU name: %s\n", devProp.name);
        // printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

        // Host allocates memories on device
        float *d_in1, *d_in2, *d_out;
        size_t nBytes = n * sizeof(float);
        CHECK(cudaMalloc(&d_in1, nBytes));
        CHECK(cudaMalloc(&d_in2, nBytes));
        CHECK(cudaMalloc(&d_out, nBytes));

        // Host copies data to device memories
        CHECK(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_in2, in2, nBytes, cudaMemcpyHostToDevice));

        // Host invokes kernel function to add vectors on device
        dim3 blockSize(512); // For simplicity, you can temporarily view blockSize as a number
        dim3 gridSize((n>>2 + blockSize.x - 1) / blockSize.x); // Similarity, view gridSize as a number
        
        if (ver == 1){
            timer.Start();
            addVecOnDeviceVersion1<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, n);

            cudaDeviceSynchronize();
            timer.Stop();

            float time = timer.Elapsed();
            printf("%25.10lf", time);

        } else{
            timer.Start();
            addVecOnDeviceVersion2<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, n);

            cudaDeviceSynchronize();
            timer.Stop();

            float time = timer.Elapsed();
            printf("%30.10lf\n", time);
        }

        // Host copies result from device memory
        CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

        // Free device memories
        CHECK(cudaFree(d_in1));
        CHECK(cudaFree(d_in2));
        CHECK(cudaFree(d_out));
        
    }
    
    float time = timer.Elapsed();
}

void evaluateTime(int n){
    float *in1, *in2; // Input vectors
    float *correctOut, *out_ver1,*out_ver2;  // Output vector

    
    // Allocate memories for in1, in2, out_ver1, out_ver2
    size_t nBytes = n * sizeof(float);
    in1 = (float *)malloc(nBytes);
    in2 = (float *)malloc(nBytes);
    correctOut = (float *)malloc(nBytes);
    out_ver1 = (float *)malloc(nBytes);
    out_ver2 = (float *)malloc(nBytes);
    
    // Input data into in1, in2
    for (int i = 0; i < n; i++)
    {
    	in1[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    	in2[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    // Add vectors (on host)
    addVec(in1, in2, correctOut, n, 0);
    
    // Add vectors on device with version1
    addVec(in1, in2, out_ver1, n, 1);
	
    // Add in1 & in2 on device
    addVec(in1, in2, out_ver2, n, 2);

    // Check correctness
    for (int i = 0; i < n; i++)
    {
    	if (out_ver1[i] =! out_ver2[i] || out_ver2[i] != correctOut[i])
    	{
            printf("INCORRECT :(\n");
            return;
    	}
    }

    printf("CORRECT :)\n");

    // free memories for in1, in2, out_ver1, out_ver2
    free(in1);
    free(in2);
    free(correctOut);
    free(out_ver1);
    free(out_ver2);
}
