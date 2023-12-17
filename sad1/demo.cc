#include <iostream>
#include "src/mnist.h"
#include "src/kernel/HW3_P1.cu"



int main(int argc, char **argv) {
	// MNIST dataset("CSC14120/sad1/data/fashion-mnist/");
	// int n_train = dataset.train_data_col;
	// int dim_in = dataset.train_data_dim.first * dataset.train_data_dim.second;
	// std::cout << "fashion-mnist train number: " << n_train << std::endl;
  	// std::cout << "fashion-mnist test number: " << dataset.test_data_col << std::endl;

	// std::cout << "train dim: " << dim_in << std::endl;
  
	// std::cout << "sad1" << std::endl;
    // return 0;
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
