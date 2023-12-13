#include "Test.cuh"

__global__ void test_kernel(void) {
}

namespace Wrapper {
	void wrapper(void)
	{
		test_kernel <<<2, 2>>> ();
		printf("Hello, world!");
	}
}