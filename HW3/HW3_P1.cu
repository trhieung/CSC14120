#include <stdio.h>
#include <stdint.h>

#define FILTER_WIDTH 9
__constant__ float dc_filter[FILTER_WIDTH * FILTER_WIDTH];

#define CHECK(call)                                                \
	{                                                              \
		const cudaError_t error = call;                            \
		if (error != cudaSuccess)                                  \
		{                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
			fprintf(stderr, "code: %d, reason: %s\n", error,       \
					cudaGetErrorString(error));                    \
			exit(EXIT_FAILURE);                                    \
		}                                                          \
	}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

void readPnm(char *fileName, int &width, int &height, uchar3 *&pixels)
{
	FILE *f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);

	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 *pixels, int width, int height, char *fileName)
{
	FILE *f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "P3\n%i\n%i\n255\n", width, height);

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);

	fclose(f);
}

__global__ void blurImgKernel1(uchar3 *inPixels, int width, int height,
							   float *filter, int filterWidth,
							   uchar3 *outPixels)
{

	// TODO
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

	if (c < width && r < height)
	{
		int i = r * width + c;
		float3 sum = make_float3(0.0f, 0.0f, 0.0f);

		for (int fy = 0; fy < filterWidth; fy++)
		{
			for (int fx = 0; fx < filterWidth; fx++)
			{
				int imageX = c - (filterWidth - 1) / 2 + fx;
				int imageY = r - (filterWidth - 1) / 2 + fy;

				uchar3 pixel;
				float filterValue = filter[fy * filterWidth + fx];
				if (imageX < 0)
				{
					if (imageY < 0)
					{
						pixel = inPixels[0 * width + 0];
					}
					else if (imageY >= height)
					{
						pixel = inPixels[(height - 1) * width + 0];
					}
					else
					{ // imageY >= 0 && imageY <height
						pixel = inPixels[imageY * width + 0];
					}
				}
				else if (imageX >= width)
				{
					if (imageY < 0)
					{
						pixel = inPixels[0 * width + width - 1];
					}
					else if (imageY >= height)
					{
						pixel = inPixels[(height - 1) * width + width - 1];
					}
					else
					{ // imageY >= 0 && imageY <height
						pixel = inPixels[imageY * width + width - 1];
					}
				}
				else
				{ // imageX >= 0 && imageX <width
					if (imageY < 0)
					{
						pixel = inPixels[0 * width + imageX];
					}
					else if (imageY >= height)
					{
						pixel = inPixels[(height - 1) * width + imageX];
					}
					else
					{ // imageY >= 0 && imageY <height
						pixel = inPixels[imageY * width + imageX];
					}
				}

				sum.x += filterValue * pixel.x;
				sum.y += filterValue * pixel.y;
				sum.z += filterValue * pixel.z;
				// Store the result in the output image
				outPixels[i] = make_uchar3((unsigned char)sum.x, (unsigned char)sum.y, (unsigned char)sum.z);
			}
		}
	}
}

__global__ void blurImgKernel2(uchar3 *inPixels, int width, int height,
							   float *filter, int filterWidth,
							   uchar3 *outPixels)
{
	// TODO
	extern __shared__ uchar3 s_inPixels[];

	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

	if (c < width && r < height)
	{
		// copy to share memory
		int s_width = blockDim.x + filterWidth - 1;
		int s_height = blockDim.y + filterWidth - 1;

		int bound_size = filterWidth / 2;
		int sc = c - bound_size;
		int sr = r - bound_size;

		int s_idx = (threadIdx.y + bound_size) * s_width + (threadIdx.x + bound_size);
		int v_idx = r * width + c;

		s_inPixels[s_idx] = inPixels[v_idx];
		// __syncthreads();
		int s_row, s_col;

		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			for (int i = 0; i < s_height; i++)
			{
				for (int j = 0; j < s_width; j++)
				{
					if (i >= bound_size && i < (s_height - bound_size) &&
						j >= bound_size && j < (s_width - bound_size))
						continue;
					else
					{
						// continue;
						s_row = sr + i;
						s_col = sc + j;

						s_idx = i * s_width + j;

						if (s_col < 0)
						{
							if (s_row < 0)
							{
								v_idx = 0 * width + 0;
							}
							else if (s_row >= height)
							{
								v_idx = (height - 1) * width + 0;
							}
							else
							{ // s_row >= 0 && s_row <height
								v_idx = s_row * width + 0;
							}
						}
						else if (s_col >= width)
						{
							if (s_row < 0)
							{
								v_idx = 0 * width + width - 1;
							}
							else if (s_row >= height)
							{
								v_idx = (height - 1) * width + width - 1;
							}
							else
							{ // s_row >= 0 && s_row <height
								v_idx = s_row * width + width - 1;
							}
						}
						else
						{ // s_col >= 0 && s_col <width
							if (s_row < 0)
							{
								v_idx = 0 * width + s_col;
							}
							else if (s_row >= height)
							{
								v_idx = (height - 1) * width + s_col;
							}
							else
							{ // s_row >= 0 && s_row <height
								v_idx = s_row * width + s_col;
							}
						}

						s_inPixels[s_idx] = inPixels[v_idx];
					}
				}
			}
		}

		__syncthreads();

		// // print blocksharedim
		// if (blockIdx.x == 0 && blockIdx.y == 31 && threadIdx.x == 0 && threadIdx.y == 0)
		// {
		// 	for (int i = 0; i < s_height; i++)
		// 	{
		// 		for (int j = 0; j < s_width; j++)
		// 		{
		// 			uchar3 pixel = s_inPixels[i * s_width + j];
		// 			printf("(%d, %d, %d) ", static_cast<int>(pixel.x), static_cast<int>(pixel.y), static_cast<int>(pixel.z));
		// 		}
		// 		printf("\n");
		// 	}
		// }

		// calculate blur
		int i = r * width + c;
		float3 sum = make_float3(0.0f, 0.0f, 0.0f);

		for (int fy = 0; fy < filterWidth; fy++)
		{
			for (int fx = 0; fx < filterWidth; fx++)
			{

				uchar3 pixel = s_inPixels[(threadIdx.y + fy) * s_width + (threadIdx.x + fx)];
				float filterValue = filter[fy * filterWidth + fx];

				sum.x += filterValue * pixel.x;
				sum.y += filterValue * pixel.y;
				sum.z += filterValue * pixel.z;
				// Store the result in the output image=
				outPixels[i] = make_uchar3((unsigned char)sum.x, (unsigned char)sum.y, (unsigned char)sum.z);
			}
		}
	}
	else
	{
		return;
	}
}

__global__ void blurImgKernel3(uchar3 *inPixels, int width, int height,
							   int filterWidth,
							   uchar3 *outPixels)
{
	// TODO
	extern __shared__ uchar3 s_inPixels[];

	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

	if (c < width && r < height)
	{
		// copy to share memory
		int s_width = blockDim.x + filterWidth - 1;
		int s_height = blockDim.y + filterWidth - 1;

		int bound_size = filterWidth / 2;
		int sc = c - bound_size;
		int sr = r - bound_size;

		int s_idx = (threadIdx.y + bound_size) * s_width + (threadIdx.x + bound_size);
		int v_idx = r * width + c;

		s_inPixels[s_idx] = inPixels[v_idx];
		// __syncthreads();

		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			for (int i = 0; i < s_height; i++)
			{
				for (int j = 0; j < s_width; j++)
				{
					if (i >= bound_size && i < (s_height - bound_size) &&
						j >= bound_size && j < (s_width - bound_size))
						continue;
					else
					{
						// continue;
						int s_row = sr + i;
						int s_col = sc + j;

						s_idx = i * s_width + j;

						if (s_col < 0)
						{
							if (s_row < 0)
							{
								v_idx = 0 * width + 0;
							}
							else if (s_row >= height)
							{
								v_idx = (height - 1) * width + 0;
							}
							else
							{ // s_row >= 0 && s_row <height
								v_idx = s_row * width + 0;
							}
						}
						else if (s_col >= width)
						{
							if (s_row < 0)
							{
								v_idx = 0 * width + width - 1;
							}
							else if (s_row >= height)
							{
								v_idx = (height - 1) * width + width - 1;
							}
							else
							{ // s_row >= 0 && s_row <height
								v_idx = s_row * width + width - 1;
							}
						}
						else
						{ // s_col >= 0 && s_col <width
							if (s_row < 0)
							{
								v_idx = 0 * width + s_col;
							}
							else if (s_row >= height)
							{
								v_idx = (height - 1) * width + s_col;
							}
							else
							{ // s_row >= 0 && s_row <height
								v_idx = s_row * width + s_col;
							}
						}

						s_inPixels[s_idx] = inPixels[v_idx];
					}
				}
			}
		}

		__syncthreads();

		// // print blocksharedim
		// if (blockIdx.x == 0 && blockIdx.y == 31 && threadIdx.x == 0 && threadIdx.y == 0)
		// {
		// 	for (int i = 0; i < s_height; i++)
		// 	{
		// 		for (int j = 0; j < s_width; j++)
		// 		{
		// 			uchar3 pixel = s_inPixels[i * s_width + j];
		// 			printf("(%d, %d, %d) ", static_cast<int>(pixel.x), static_cast<int>(pixel.y), static_cast<int>(pixel.z));
		// 		}
		// 		printf("\n");
		// 	}
		// }

		// calculate blur
		int i = r * width + c;
		float3 sum = make_float3(0.0f, 0.0f, 0.0f);

		for (int fy = 0; fy < filterWidth; fy++)
		{
			for (int fx = 0; fx < filterWidth; fx++)
			{

				uchar3 pixel = s_inPixels[(threadIdx.y + fy) * s_width + (threadIdx.x + fx)];
				float filterValue = dc_filter[fy * filterWidth + fx];

				sum.x += filterValue * pixel.x;
				sum.y += filterValue * pixel.y;
				sum.z += filterValue * pixel.z;
				// Store the result in the output image=
				outPixels[i] = make_uchar3((unsigned char)sum.x, (unsigned char)sum.y, (unsigned char)sum.z);
			}
		}
	}
	else
	{
		return;
	}
}

void blurImg(uchar3 *inPixels, int width, int height, float *filter, int filterWidth,
			 uchar3 *outPixels,
			 bool useDevice = false, dim3 blockSize = dim3(1, 1), int kernelType = 1)
{
	if (useDevice == false)
	{
		// TODO
		for (int r = 0; r < height; r++)
		{
			for (int c = 0; c < width; c++)
			{
				int i = r * width + c;
				float3 sum = make_float3(0.0f, 0.0f, 0.0f);

				for (int fy = 0; fy < filterWidth; fy++)
				{
					for (int fx = 0; fx < filterWidth; fx++)
					{
						int imageX = c - (filterWidth - 1) / 2 + fx;
						int imageY = r - (filterWidth - 1) / 2 + fy;

						uchar3 pixel;
						float filterValue = filter[fy * filterWidth + fx];
						if (imageX < 0)
						{
							if (imageY < 0)
							{
								pixel = inPixels[0 * width + 0];
							}
							else if (imageY >= height)
							{
								pixel = inPixels[(height - 1) * width + 0];
							}
							else
							{ // imageY >= 0 && imageY <height
								pixel = inPixels[imageY * width + 0];
							}
						}
						else if (imageX >= width)
						{
							if (imageY < 0)
							{
								pixel = inPixels[0 * width + width - 1];
							}
							else if (imageY >= height)
							{
								pixel = inPixels[(height - 1) * width + width - 1];
							}
							else
							{ // imageY >= 0 && imageY <height
								pixel = inPixels[imageY * width + width - 1];
							}
						}
						else
						{ // imageX >= 0 && imageX <width
							if (imageY < 0)
							{
								pixel = inPixels[0 * width + imageX];
							}
							else if (imageY >= height)
							{
								pixel = inPixels[(height - 1) * width + imageX];
							}
							else
							{ // imageY >= 0 && imageY <height
								pixel = inPixels[imageY * width + imageX];
							}
						}

						sum.x += filterValue * pixel.x;
						sum.y += filterValue * pixel.y;
						sum.z += filterValue * pixel.z;
						// Store the result in the output image
						outPixels[i] = make_uchar3((unsigned char)sum.x, (unsigned char)sum.y, (unsigned char)sum.z);
					}
				}
			}
		}
	}
	else // Use device
	{
		GpuTimer timer;

		printf("\nKernel %i, ", kernelType);
		// Allocate device memories
		uchar3 *d_inPixels, *d_outPixels;
		float *d_filter;
		size_t pixelsSize = width * height * sizeof(uchar3);
		size_t filterSize = filterWidth * filterWidth * sizeof(float);
		CHECK(cudaMalloc(&d_inPixels, pixelsSize));
		CHECK(cudaMalloc(&d_outPixels, pixelsSize));
		if (kernelType == 1 || kernelType == 2)
		{
			CHECK(cudaMalloc(&d_filter, filterSize));
		}

		// Copy data to device memories
		CHECK(cudaMemcpy(d_inPixels, inPixels, pixelsSize, cudaMemcpyHostToDevice));
		if (kernelType == 1 || kernelType == 2)
		{
			CHECK(cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice));
		}
		else
		{
			// TODO: copy data from "filter" (on host) to "dc_filter" (on CMEM of device)
			CHECK(cudaMemcpyToSymbol(dc_filter, filter, filterSize));
		}

		// Call kernel
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
		printf("block size %ix%i, grid size %ix%i\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);
		timer.Start();
		if (kernelType == 1)
		{
			// TODO: call blurImgKernel1
			blurImgKernel1<<<gridSize, blockSize>>>(d_inPixels, width, height, d_filter, filterWidth, d_outPixels);
		}
		else if (kernelType == 2)
		{
			// TODO: call blurImgKernel2
			size_t sharedMemorySize = (blockSize.x + filterWidth - 1) * (blockSize.y + filterWidth - 1) * sizeof(uchar3);
			blurImgKernel2<<<gridSize, blockSize, sharedMemorySize>>>(d_inPixels, width, height, d_filter, filterWidth, d_outPixels);
		}
		else
		{
			// TODO: call blurImgKernel3
			size_t sharedMemorySize = (blockSize.x + filterWidth - 1) * (blockSize.y + filterWidth - 1) * sizeof(uchar3);
			blurImgKernel3<<<gridSize, blockSize, sharedMemorySize>>>(d_inPixels, width, height, filterWidth, d_outPixels);
		}
		timer.Stop();
		float time = timer.Elapsed();
		printf("Kernel time: %f ms\n", time);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		// Copy result from device memory
		CHECK(cudaMemcpy(outPixels, d_outPixels, pixelsSize, cudaMemcpyDeviceToHost));

		// Free device memories
		CHECK(cudaFree(d_inPixels));
		CHECK(cudaFree(d_outPixels));
		if (kernelType == 1 || kernelType == 2)
		{
			CHECK(cudaFree(d_filter));
		}
	}
}

float computeError(uchar3 *a1, uchar3 *a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

void printError(uchar3 *deviceResult, uchar3 *hostResult, int width, int height)
{
	float err = computeError(deviceResult, hostResult, width * height);
	printf("Error: %f\n", err);
}

char *concatStr(const char *s1, const char *s2)
{
	char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
	CHECK(cudaGetDeviceProperties(&devProv, 0));
	printf("**********GPU info**********\n");
	printf("Name: %s\n", devProv.name);
	printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
	printf("Num SMs: %d\n", devProv.multiProcessorCount);
	printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
	printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
	printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
	printf("CMEM: %lu bytes\n", devProv.totalConstMem);
	printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
	printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
	printf("****************************\n");
}

int main(int argc, char **argv)
{
	if (argc != 3 && argc != 5)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	printDeviceInfo();

	// Read input image file
	int width, height;
	uchar3 *inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

	// Set up a simple filter with blurring effect
	int filterWidth = FILTER_WIDTH;
	float *filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
	for (int filterR = 0; filterR < filterWidth; filterR++)
	{
		for (int filterC = 0; filterC < filterWidth; filterC++)
		{
			filter[filterR * filterWidth + filterC] = 1. / (filterWidth * filterWidth);
		}
	}

	// Blur input image not using device
	uchar3 *correctOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, correctOutPixels);

	// Blur input image using device, kernel 1
	dim3 blockSize(16, 16); // Default
	if (argc == 5)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}
	uchar3 *outPixels1 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, outPixels1, true, blockSize, 1);
	printError(outPixels1, correctOutPixels, width, height);

	// Blur input image using device, kernel 2
	uchar3 *outPixels2 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, outPixels2, true, blockSize, 2);
	printError(outPixels2, correctOutPixels, width, height);

	// Blur input image using device, kernel 3
	uchar3 *outPixels3 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, outPixels3, true, blockSize, 3);
	printError(outPixels3, correctOutPixels, width, height);

	// Write results to files
	char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(correctOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(outPixels1, width, height, concatStr(outFileNameBase, "_device1.pnm"));
	writePnm(outPixels2, width, height, concatStr(outFileNameBase, "_device2.pnm"));
	writePnm(outPixels3, width, height, concatStr(outFileNameBase, "_device3.pnm"));

	// Free memories
	free(inPixels);
	free(filter);
	free(correctOutPixels);
	free(outPixels1);
	free(outPixels2);
	free(outPixels3);
}
