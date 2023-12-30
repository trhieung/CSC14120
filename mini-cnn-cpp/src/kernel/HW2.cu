#include "./HW2.cuh"

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num blocks per SM: %d\n", devProv.maxBlocksPerMultiProcessor); 
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");

}

// P1
__global__ void reduceBlksKernel1(int * in, int * out, int n)
{
	int i = blockIdx.x * 2 * blockDim.x + 2 * threadIdx.x;
    for (int stride = 1; stride <= blockDim.x; stride <<= 1){
        if (threadIdx.x % stride == 0)
            if (i + stride < n)
                in[i] += in[i + stride];
        __syncthreads(); // Synchronize within each block
    }
    if (threadIdx.x == 0)
        atomicAdd(out, in[blockIdx.x * 2 * blockDim.x]);
}

__global__ void reduceBlksKernel2(int * in, int * out,int n)
{
	// TODO
    int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
    {
        int i = numElemsBeforeBlk + threadIdx.x * 2 * stride;
        if (threadIdx.x < blockDim.x / stride)
            if (i + stride < n)
                in[i] += in[i + stride];
        __syncthreads(); // Synchronize within each block
    }
    if (threadIdx.x == 0){
        atomicAdd(out, in[numElemsBeforeBlk]);
    }
}

__global__ void reduceBlksKernel3(int * in, int * out,int n)
{
	// TODO
    int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    for(int stride = blockDim.x; stride > 0; stride >>= 1){
        if(threadIdx.x < stride)
            in[i] += in[i+stride];
        __syncthreads();
    }
    
    if (threadIdx.x == 0)
        atomicAdd(out, in[blockIdx.x * 2 * blockDim.x]);
}

int reduce(int const * in, int n,
        bool useDevice, dim3 blockSize, int kernelType)
{

	GpuTimer timer;
	int result = 0; // Init
	if (useDevice == false)
	{
		timer.Start();
		result = in[0];
		for (int i = 1; i < n; i++)
		{
			result += in[i];
		}
		timer.Stop();
		float hostTime = timer.Elapsed();
		printf("Host time: %f ms\n",hostTime);
        printf("correct result: %d \n",result);
	}
	else // Use device
	{
		// Allocate device memories
		int * d_in, * d_out;
		dim3 gridSize(((n - 1)>>1) / blockSize.x + 1); // TODO: Compute gridSize from n and blockSize
		size_t nBytes = n * sizeof(int);

		// TODO: Allocate device memories
        CHECK(cudaMalloc(&d_in, nBytes));
        CHECK(cudaMalloc(&d_out, sizeof(int)));

		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));
        
		// Call kernel
		timer.Start();
		if (kernelType == 1)
			reduceBlksKernel1<<<gridSize, blockSize>>>(d_in, d_out, n);
		else if (kernelType == 2)
			reduceBlksKernel2<<<gridSize, blockSize>>>(d_in, d_out, n);
		else 
			reduceBlksKernel3<<<gridSize, blockSize>>>(d_in, d_out, n);

		cudaDeviceSynchronize();
		timer.Stop();
		float kernelTime = timer.Elapsed();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost));

		// TODO: Free device memories
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
		// Print info
		printf("\nKernel %d\n", kernelType);
		printf("Grid size: %d, block size: %d\n", gridSize.x, blockSize.x);
		printf("Kernel time = %f ms\n", kernelTime);
        printf("Kernel result: %d \n",result);
	}

	return result;
}
void HW2_P1_checkCorrectness(int r1, int r2)
{
	if (r1 == r2)
		printf("CORRECT :)\n");
	else
		printf("INCORRECT :(\n");
}

void HW2_P1(){
    printDeviceInfo();

	// Set up input size
    int n = (1 << 24)+1;
    printf("Input size: %d\n", n);

    // Set up input data
    int * in = (int *) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        // Generate a random integer in [0, 255]
        in[i] = (int)(rand() & 0xFF);
    }

    // Reduce NOT using device
    int correctResult = reduce(in, n);

    // Reduce using device, kernel1
    dim3 blockSize(1024); // Default
    // if (argc == 2)
    // 	blockSize.x = atoi(argv[1]); 
 	
	int result1 = reduce(in, n, true, blockSize, 1);
    HW2_P1_checkCorrectness(result1, correctResult);

    // Reduce using device, kernel2
    int result2 = reduce(in, n, true, blockSize, 2);
    HW2_P1_checkCorrectness(result2, correctResult);

    // Reduce using device, kernel3
    int result3 = reduce(in, n, true, blockSize, 3);
    HW2_P1_checkCorrectness(result3, correctResult);

    // Free memories
    free(in);
}

// P2
__global__ void matrix_multiplication_kernel1(float* A, float* B, float* C, int m, int n, int k)
{
	//TODO
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float val = 0.0;
        for (int i = 0; i < n; ++i) {
            val += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = val;
    }
}

__global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k)
{
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	//TODO

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;  

    float val = 0.0f;

    for (int i = 0; i < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++i) {
        if (row < m && i * TILE_WIDTH + tx < n) {
            s_A[ty][tx] = A[row * n + i * TILE_WIDTH + tx];
        } else {
            s_A[ty][tx] = 0.0;
        }

        if (i * TILE_WIDTH + ty < n && col < k) {
            s_B[ty][tx] = B[(i * TILE_WIDTH + ty) * k + col];
        } else {
            s_B[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j) {
            val += s_A[ty][j] * s_B[j][tx];
        }

        __syncthreads();
    }

    if (row < m && col < k) {
        C[row * k + col] = val;
    }  
}

void matrix_multiplication(float* A, float* B, float* C, int m, int n, int k,
    bool useDevice, dim3 blockSize,int kernelType)
{
    GpuTimer timer;
    timer.Start();
    if (useDevice == false)
    {
        // TODO
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k; ++j) {
                float val = 0.0;
                for (int x = 0; x < n; ++x) {
                    val += A[i * n + x] * B[x * k + j];
                }
                C[i * k + j] = val;
            }
        }
    }
    else // Use device
    {
        // TODO: Allocate device memories
        float* d_A, * d_B, * d_C;
        size_t size_A = m * n * sizeof(float);
        size_t size_B = n * k * sizeof(float);
        size_t size_C = m * k * sizeof(float);

        CHECK(cudaMalloc((void**)&d_A, size_A));
        CHECK(cudaMalloc((void**)&d_B, size_B));
        CHECK(cudaMalloc((void**)&d_C, size_C));

        // Copy data to device memories
        CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
        
        dim3 gridSize((k - 1) / blockSize.x + 1, (m - 1) / blockSize.y + 1); // TODO: Compute gridSize
        
		if (kernelType == 1)
			matrix_multiplication_kernel1<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
		else if (kernelType == 2)
			matrix_multiplication_kernel2<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

		CHECK(cudaGetLastError());
        // TODO: Copy result from device memory
        CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

        // TODO: Free device memories
        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));
        
		printf("Grid size: %d * %d, block size: %d * %d\n", 
			gridSize.x,gridSize.y, blockSize.x,blockSize.y);
    }
    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (%s): %f ms\n",
        useDevice == true ? "use device" : "use host", time);
}

float HW2_P2_checkCorrectness(float * a1, float* a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)	
		err += abs(a1[i] - a2[i]);
	err /= n;
	return err;
}

void HW2_P2(){
    printDeviceInfo();
	
	//Declare variables
    float* h_A; // The A matrix
    float* h_B; // The B matrix
    float* h_C; // The output C matrix
    float* correct_C; // The output C matrix

    int m;    // number of rows in the matrix A
    int n; // number of columns in the matrix A, number of rows in the matrix B
    int k; // number of columns in the matrix B

    m = (1 << 10);
    n = (1 << 9);
    k = (1 << 10);

    // Set up input data
    h_A = (float*)malloc(m * n * sizeof(float));
    h_B = (float*)malloc(n * k * sizeof(float));
    h_C = (float*)malloc(m * k * sizeof(float));
    correct_C = (float*)malloc(m * k * sizeof(float));

    for (int i = 0; i < m; i++)
        for (int j = 0;j < n;j++)
            h_A[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
 
    for (int i = 0; i < n; i++)
        for (int j = 0;j < k;j++)
            h_B[i*k+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);


    // Add vectors (on host)
    matrix_multiplication(h_A,h_B,correct_C,m,n,k);
	printf("\n");

	dim3 blockSize(32, 32); // Default
	// if (argc == 3)
	// {
	// 	blockSize.x = atoi(argv[1]);
	// 	blockSize.y = atoi(argv[2]);
	// } 
    // Add in1 & in2 on device
	printf("Basic Matrix Multiplication:\n");
    matrix_multiplication(h_A, h_B, h_C, m, n, k, true,blockSize,1);
	float err = HW2_P2_checkCorrectness(h_C, correct_C,m*k);
	printf("Error between device result and host result: %f\n\n", err);

	printf("Shared memory Matrix Multiplication:\n");
    matrix_multiplication(h_A, h_B, h_C, m, n, k, true,blockSize,2);
	err = HW2_P2_checkCorrectness(h_C, correct_C,m*k);
	printf("Error between device result and host result: %f", err);	
	
    free(h_A);
    free(h_B);
    free(h_C);
    free(correct_C);
}

