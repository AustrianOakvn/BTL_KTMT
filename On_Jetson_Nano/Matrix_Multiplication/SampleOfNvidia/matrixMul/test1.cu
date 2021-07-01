/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

using std::chrono::steady_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

#define FILE_NAME "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Lab2_MuliMatrix/SampleOfNvidia/matrixMul/benchmark_log_JetsonNano_shmem.txt"
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
                                                        float *B, int wA,
                                                        int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = (rand() / (float)RAND_MAX) * 100.0;
}

void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv,
                   int block_size, const dim3 &dimsA,
                   const dim3 &dimsB, std::ostream &fileout) 
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float*)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float*)malloc(mem_size_B);
   //  cudaStream_t stream;

    // Initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int size_C = dimsC.x * dimsC.y;
    unsigned int mem_size_C = size_C * sizeof(float);
    float *h_C = (float*)malloc(mem_size_C);

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc((void **)(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc((void **)(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc((void **)(&d_C), mem_size_C));
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

   //  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));


    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
   //  printf("Computing result using CUDA Kernel...\n");

   //  // Performs warmup operation using matrixMul CUDA kernel
   //  if (block_size == 16) {
   //      MatrixMulCUDA<16> <<< grid, threads, 0>>>(d_C, d_A, d_B,
   //                                              dimsA.x, dimsB.x);
   //  } else {
   //      MatrixMulCUDA<32> <<< grid, threads, 0>>>(d_C, d_A, d_B,
   //                                              dimsA.x, dimsB.x);
   //  }

   // printf("done\n");
   // //  checkCudaErrors(cudaStreamSynchronize(stream));
   //  checkCudaErrors(cudaDeviceSynchronize());

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Execute the kernel
    int nIter = 50;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16> <<<grid, threads, 0>>>(d_C, d_A, d_B,
                                                    dimsA.x, dimsB.x);
        } else {
            MatrixMulCUDA<32> <<<grid, threads, 0>>>(d_C, d_A, d_B,
                                                    dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                               static_cast<double>(dimsA.y) *
                               static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
                       (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops," \
        " WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);
    
    fileout << (int)dimsA.x << ", " << msecPerMatrixMul << ", " << flopsPerMatrixMul << ", " << gigaFlops;


    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    
    // verify the result of matrix multiplication
    float *reference = (float *)malloc(mem_size_C);
    steady_clock::time_point start_CPU = steady_clock::now();
    matrixMulCPU(reference, h_A, h_B, (unsigned int)dimsA.y, (unsigned int)dimsA.x, (unsigned int)dimsB.x);   // matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    steady_clock::time_point end_CPU = steady_clock::now();
    fileout << ", " << duration_cast <duration<double>>(end_CPU - start_CPU).count() << "\n";
    printf("done.\n");

    printf("Checking computed result for correctness: ");
    bool correct = sdkCompareL2fe(reference, h_C, size_C, 1.0e-6f);
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    // printf("\nNOTE: The CUDA Samples are not meant for performance"\
    //        "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv) {
    std::ofstream fileout;
    fileout.open(FILE_NAME, std::ios_base::out | std::ios_base::app );
    fileout << "kernel_size, time(msec), ops, GFlop/s, time_CPU(sec)\n" ;
    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    // int dev = findCudaDevice(argc, (const char **)argv);
    int block_size = 32;

    // for (int i = 1; i <= 128; i *= 2)
    int i = 128;
    {
        dim3 dimsA(i * block_size, i * block_size, 1);
        dim3 dimsB(i * block_size, i * block_size, 1);
        if (dimsA.x != dimsB.y) {
            printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
                   dimsA.x, dimsB.y);
            exit(EXIT_FAILURE);
        }
    
        printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
                                                   dimsB.x, dimsB.y);
    
        int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, fileout);
        
        if (matrix_result != 0)
            return matrix_result;
    }

    fileout.close();
    return 0;
}

