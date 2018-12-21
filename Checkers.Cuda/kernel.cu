
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
//#include <cutil.h>		// timers

// cuda kernel (internal)
__global__ void some_calculations(float *a, unsigned int N, unsigned int M)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		// note1: no need for shared memory here
		// note2: global memory access is coalesced
		//        (no structs, float only used)

		// do computations M times on each thread
		// to extend processor time
		for (unsigned int i = 0; i < M; i++)
		{
			// some easy arithmetics		
			a[idx] = a[idx] * a[idx] * 0.1 - a[idx] - 10;
		}
	}
}

// internal variable (example, not really necessary here)
static volatile int PRINT_ERRORS = 1;	// true

// cuda helper function (internal)
int checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		if (PRINT_ERRORS)
			printf("Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		return err;
	}
	return 0; // cudaSuccess
}

// external variable example
extern "C" { float __declspec(dllexport) sExecutionTime = -1; }

// variable wrapper function
extern "C" float __declspec(dllexport) __stdcall GetExecutionTime()
{
	return 123;
}

// cuda wrapper function
extern "C" int __declspec(dllexport) __stdcall SomeCalculationsCU
(
	float *a_h,							// pointer to input array
	const unsigned int N,				// input array size
	const unsigned int M,				// kernel M parameter
	const int cuBlockSize = 512,		// kernel block size (max 512)
	const int showErrors = 1			// show CUDA errors in console window
)
{
	int tmp = PRINT_ERRORS;
	PRINT_ERRORS = showErrors;

	float *a_d;							// pointer to device array
	size_t size = N * sizeof(float);
	int cuerr = 0;						// no errors
	unsigned int timer = 0;

	cudaMalloc((void**)&a_d, size);		// allocate array on device    
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

	int n_blocks = N / cuBlockSize + (N % cuBlockSize == 0 ? 0 : 1);

	//cutCreateTimer(&timer);			    // from cutil.h
	//cutStartTimer(timer);
	some_calculations << <n_blocks, cuBlockSize >> > (a_d, N, M);	// kernel invocation
	cudaThreadSynchronize();			// by default kernel runs in parallel with CPU code
	//cutStopTimer(timer);

	cuerr = checkCUDAError("cuda kernel");

	cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
	if (!cuerr) cuerr = checkCUDAError("cuda memcpy");

	//sExecutionTime = cutGetTimerValue(timer);

	cudaFree(a_d);
	if (!cuerr) cuerr = checkCUDAError("cuda free");

	PRINT_ERRORS = tmp;
	return cuerr;
}

// cpu version for comparison
extern "C" void __declspec(dllexport) __stdcall SomeCalculationsCPU
(
	float *a_h,
	const unsigned int N,
	const unsigned int M
)
{
	unsigned int timer = 0;
	//cutCreateTimer(&timer);
	//cutStartTimer(timer);
	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < M; j++)
			*(a_h + i) = *(a_h + i) * *(a_h + i) * 0.1 - *(a_h + i) - 10;
	//cutStopTimer(timer);
	//sExecutionTime = cutGetTimerValue(timer);
}
