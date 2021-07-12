#include <cstdlib>
#include "cstdlib"
#include <process.h>
#include <malloc.h>
#include <ctime>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "Bitmap.h"
#include <device_functions.h>
using namespace std;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define MEDIAN_DIMENSION 3
#define MEDIAN_LENGTH 9

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

__global__ void medianFilter_gpu(unsigned char* Input_Image, unsigned char *Output_Image, int Image_Width, int Image_Height) {
	const int tx_l = threadIdx.x;
	const int ty_l = threadIdx.y;

	const int tx_g = blockIdx.x * blockDim.y + tx_l;
	const int ty_g = blockIdx.y * blockDim.y + ty_l;

	__shared__ unsigned char smem[BLOCK_WIDTH + 2][BLOCK_HEIGHT + 2];

	if (tx_l == 0) smem[tx_l][ty_l + 1] = 0;
	else if (tx_l == BLOCK_WIDTH - 1) smem[tx_l + 2][ty_l + 1] = 0;
	if (ty_l == 0) {
		smem[tx_l + 1][ty_l] = 0;
		if (tx_l == 0) smem[tx_l][tx_l] = 0;
		else if (tx_l == BLOCK_WIDTH - 1) smem[tx_l + 2][ty_l] = 0;

	}
	else if (tx_l == BLOCK_HEIGHT - 1) {
		smem[tx_l + 1][ty_l + 2] = 0;
		if (tx_l == 0) smem[tx_l][ty_l + 2] = 0;
		else if (tx_l == BLOCK_WIDTH - 1) smem[tx_l + 2][ty_l + 2] = 0;
	}

	if ((tx_l == 0) && ((tx_g > 0)))                                      
		smem[tx_l][ty_l + 1] = Input_Image[ty_g * Image_Width + tx_g - 1];      // --- left border
	else if ((tx_l == BLOCK_WIDTH - 1) && (tx_g < Image_Width - 1))         
		smem[tx_l + 2][ty_l + 1] = Input_Image[ty_g * Image_Width + tx_g + 1];      // --- right border
	if ((ty_l == 0) && (ty_g > 0)) {
		smem[tx_l + 1][ty_l] = Input_Image[(ty_g - 1) * Image_Width + tx_g];    // --- upper border
		if ((tx_l == 0) && ((tx_g > 0)))                                  
			smem[tx_l][ty_l] = Input_Image[(ty_g - 1) * Image_Width + tx_g - 1];  // --- top-left corner
		else if ((tx_l == BLOCK_WIDTH - 1) && (tx_g < Image_Width - 1))     
			smem[tx_l + 2][ty_l] = Input_Image[(ty_g - 1) * Image_Width + tx_g + 1];  // --- top-right corner
	}
	else if ((ty_l == BLOCK_HEIGHT - 1) && (ty_g < Image_Height - 1)) {
		smem[tx_l + 1][ty_l + 2] = Input_Image[(ty_g + 1) * Image_Width + tx_g];    // --- bottom border
		if ((tx_l == 0) && ((tx_g > 0)))                                 
			smem[tx_l][ty_l + 2] = Input_Image[(ty_g - 1) * Image_Width + tx_g - 1];  // --- bottom-left corder
		else if ((tx_l == BLOCK_WIDTH - 1) && (tx_g < Image_Width - 1))     
			smem[tx_l + 2][ty_l + 2] = Input_Image[(ty_g + 1) * Image_Width + tx_g + 1];  // --- bottom-right corner
	}
	__syncthreads();
	unsigned char v[9] = { smem[tx_l][ty_l], smem[tx_l + 1][ty_l], smem[tx_l + 2][ty_l],
						  smem[tx_l][ty_l + 1], smem[tx_l + 1][ty_l + 1], smem[tx_l + 2][ty_l + 1],
						  smem[tx_l][ty_l + 2], smem[tx_l + 1][ty_l + 2], smem[tx_l + 2][ty_l + 2] };

	// Buble sort
	for (int i = 0; i < 6; i++) {
		for (int j = i + 1; j < 9; j++) {
			if (v[i] = v[j]) {
				unsigned char tmp = v[i];
				v[i] = v[j];
				v[j] = tmp;
			}
		}
	}
	Output_Image[ty_g * Image_Width + tx_g] = v[4];
}

//__global__ void Original_Kernel_Function(unsigned char* Input_Image, unsigned char* Output_Image, int Image_Width, int Image_Height) {
//	__shared__ unsigned short surround[BLOCK_WIDTH * BLOCK_HEIGHT][9];
//
//	int iterator;
//
//	const int x = blockDim.x * blockIdx.x + threadIdx.x;
//	const int y = blockDim.y * blockIdx.y + threadIdx.y;
//	const int tid = threadIdx.y * blockDim.x + threadIdx.x;
//
//	if ((x >= (Image_Width - 1)) || (y >= Image_Height - 1) || (x == 0) || (y == 0)) return;
//
//	// --- Fill shared memory
//	iterator = 0;
//	for (int r = x - 1; r <= x + 1; r++) {
//		for (int c = y - 1; c <= y + 1; c++) {
//			surround[tid][iterator] = Input_Image[c * Image_Width + r];
//			iterator++;
//		}
//	}
//
//	// --- Sort shared memory to find the median using Bubble Short
//	for (int i = 0; i < 5; ++i) {
//
//		// --- Find the position of the minimum element
//		int minval = i;
//		for (int l = i + 1; l < 9; ++l) if (surround[tid][l] < surround[tid][minval]) minval = l;
//
//		// --- Put found minimum element in its place
//		unsigned short temp = surround[tid][i];
//		surround[tid][i] = surround[tid][minval];
//		surround[tid][minval] = temp;
//	}
//
//	// --- Pick the middle one
//	Output_Image[(y * Image_Width) + x] = surround[tid][4];
//
//	__syncthreads();
//
//}

__global__ void Kernel_Function_no_shared(unsigned char* Input_Image, unsigned char* Output_Image, int Image_Width, int Image_Height) {
	unsigned short surround[9];

	int iterator;

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int tid = threadIdx.y * blockDim.x + threadIdx.x;

	if ((x >= (Image_Width - 1)) || (y >= Image_Height - 1) || (x == 0) || (y == 0)) return;

	// --- Fill array private to the threads
	iterator = 0;
	for (int r = x - 1; r <= x + 1; r++) {
		for (int c = y - 1; c <= y + 1; c++) {
			surround[iterator] = Input_Image[c * Image_Width + r];
			iterator++;
		}
	}

	// --- Sort private array to find the median using Bubble Short
	for (int i = 0; i < 5; ++i) {

		// --- Find the position of the minimum element
		int minval = i;
		for (int l = i + 1; l < 9; ++l) if (surround[l] < surround[minval]) minval = l;

		// --- Put found minimum element in its place
		unsigned short temp = surround[i];
		surround[i] = surround[minval];
		surround[minval] = temp;
	}

	// --- Pick the middle one
	Output_Image[(y * Image_Width) + x] = surround[4];
}

__global__ void resizeable_kernel(unsigned char* Input_Image, unsigned char* Output_Image, int Image_Width, int Image_Height) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	__shared__ unsigned short surround[BLOCK_WIDTH * BLOCK_HEIGHT][MEDIAN_LENGTH];
	int iterator;
	const int Haft_Of_MEDIAN_LENGTH = (MEDIAN_LENGTH) / 2 + 1;
	int StartPoint = MEDIAN_DIMENSION / 2;
	int EndPoint = StartPoint + 1;
	
	const int tid = threadIdx.y * blockDim.y + threadIdx.x;

	if (x == 0 || x == Image_Width - StartPoint || y == 0 || y == Image_Height - StartPoint) {

	}
	else {
		iterator = 0;
		for (int r = x - StartPoint; r < x + EndPoint;r++) {
			for (int c = y - StartPoint;c < y + (EndPoint); c++) {
				surround[tid][iterator] = *(Input_Image + (c * Image_Width) + r);
				iterator++;
			}
		}
		for (int i = 0; i < Haft_Of_MEDIAN_LENGTH; ++i) {
			int min = i;
			for (int l = i + 1; l < MEDIAN_LENGTH; l++)
				if (surround[tid][l] < surround[tid][min])
					min = l;
			unsigned char tmp = surround[tid][i];
			surround[tid][i] = surround[tid][min];
			surround[tid][min] = tmp;
		}

		*(Output_Image + (y * Image_Width) + x) = surround[tid][Haft_Of_MEDIAN_LENGTH - 1];
		__syncthreads();
	}
}

int main() {
	//char image_path[] = "C:\\Users\\AustrianOak\\Pictures\\test_2.bmp";
	//char output_path[] = "D:\\pythonProject\\Salt_Peper_Noise\\example.bmp";
	// init bitmap image
	Bitmap* originalImage = new Bitmap();
	Bitmap* resultImageCPU = new Bitmap();
	Bitmap* resultImageGPU = new Bitmap();
	Bitmap* resultImageSharedGPU = new Bitmap();
	originalImage->Load("D:\\Cppdev\\MedianFilter\\lena512.bmp");
	//resultImageCPU->Load("D:\\Cppdev\\MedianFilter\\lena512.bmp");
	resultImageGPU->Load("D:\\Cppdev\\MedianFilter\\lena512.bmp");
	resultImageSharedGPU->Load("D:\\Cppdev\\MedianFilter\\lena512.bmp");
	cudaError_t status;
	
	////////////////////////////////////////////////
	int img_width = originalImage->Width();
	int img_height = originalImage->Height();
	
	int size = img_width * img_height * sizeof(char);
	unsigned char* deviceinputimage;
	cudaMalloc((void**)&deviceinputimage, size);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed for cudaMalloc : " << cudaGetErrorString(status) <<
			std::endl;
		return false;
	}
	cudaMemcpy(deviceinputimage, originalImage->image, size, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed for cudaMemcpy cudaMemcpyHostToDevice: " << cudaGetErrorString(status) <<
			std::endl;
		cudaFree(deviceinputimage);
		return false;
	}
	unsigned char* deviceOutputImage;
	cudaMalloc((void**)&deviceOutputImage, size);

	///////////////////////////////////////////////
	const dim3 grid(iDivUp(img_width, BLOCK_WIDTH), iDivUp(img_height, BLOCK_HEIGHT), 1);
	const dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT, 1);

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	//cudaFuncSetCacheConfig(Original_Kernel_Function, cudaFuncCachePreferShared);
	//Original_Kernel_Function << <grid, block >> > (deviceinputimage, deviceOutputImage, img_width, img_height);
	resizeable_kernel <<< grid, block >>> (deviceinputimage, deviceOutputImage, img_width, img_height);
	cudaPeekAtLastError();
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "Original kernel function time:" << time << "ms" << endl;

	cudaMemcpy(resultImageGPU->image, deviceOutputImage, size, cudaMemcpyDeviceToHost);
	status = cudaGetLastError();

	if (status != cudaSuccess) {
		cout << "Kernel failed" << endl;
		cudaFree(deviceinputimage);
		cudaFree(deviceOutputImage);
		return 1;
	}
	resultImageGPU->Save("D:\\pythonProject\\Salt_Peper_Noise\\example.bmp");
	cudaFree(deviceinputimage);
	cudaFree(deviceOutputImage);
	return 0;
	
}