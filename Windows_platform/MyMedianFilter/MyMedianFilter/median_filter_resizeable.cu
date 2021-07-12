//#include "cuda_runtime.h"
//#include "cuda_runtime_api.h"
//#include "device_launch_parameters.h"
//#include "cuda.h"
//#include <iostream>  
//#include <fstream>   
//#include <iomanip>   
//#include <windows.h>
//#include <io.h>                  
//#include <stdio.h>
//#include<conio.h>
//#include <cstdlib>
//#include "cstdlib"
//#include <process.h>
//#include <stdlib.h>
//#include <malloc.h>
//#include <ctime>
//using namespace std;
//
//#define MEDIAN_DIMENSION  3 // For matrix of 3 x 3. We can Use 5 x 5 , 7 x 7 , 9 x 9......   
//#define MEDIAN_LENGTH 9   // Shoul be  MEDIAN_DIMENSION x MEDIAN_DIMENSION = 3 x 3
//
//#define BLOCK_WIDTH 16  // Should be 8 If matrix is of larger then of 5 x 5 elese error occur as " uses too much shared data "  at surround[BLOCK_WIDTH*BLOCK_HEIGHT][MEDIAN_LENGTH]
//#define BLOCK_HEIGHT 16// Should be 8 If matrix is of larger then of 5 x 5 elese error occur as " uses too much shared data "  at surround[BLOCK_WIDTH*BLOCK_HEIGHT][MEDIAN_LENGTH]
//
//__global__ void MedianFilter_gpu(unsigned short* Device_ImageData, int Image_Width, int Image_Height) {
//
//    __shared__ unsigned short surround[BLOCK_WIDTH * BLOCK_HEIGHT][MEDIAN_LENGTH];
//
//    int iterator;
//    const int Half_Of_MEDIAN_LENGTH = (MEDIAN_LENGTH / 2) + 1;
//    int StartPoint = MEDIAN_DIMENSION / 2;
//    int EndPoint = StartPoint + 1;
//
//    const int x = blockDim.x * blockIdx.x + threadIdx.x;
//    const int y = blockDim.y * blockIdx.y + threadIdx.y;
//
//    const int tid = threadIdx.y * blockDim.y + threadIdx.x;
//
//    if (x >= Image_Width || y >= Image_Height)
//        return;
//
//    //Fill surround with pixel value of Image in Matrix Pettern of MEDIAN_DIMENSION x MEDIAN_DIMENSION
//    if (x == 0 || x == Image_Width - StartPoint || y == 0
//        || y == Image_Height - StartPoint) {
//    }
//    else {
//        iterator = 0;
//        for (int r = x - StartPoint; r < x + (EndPoint); r++) {
//            for (int c = y - StartPoint; c < y + (EndPoint); c++) {
//                surround[tid][iterator] = *(Device_ImageData + (c * Image_Width) + r);
//                iterator++;
//            }
//        }
//        //Sort the Surround Array to Find Median. Use Bubble Short  if Matrix oF 3 x 3 Matrix 
//                            //You can use Insertion commented below to Short Bigger Dimension Matrix  
//
//                                      ////      bubble short //
//
//        for (int i = 0; i < Half_Of_MEDIAN_LENGTH; ++i)
//        {
//            // Find position of minimum element
//            int min = i;
//            for (int l = i + 1; l < MEDIAN_LENGTH; ++l)
//                if (surround[tid][l] < surround[tid][min])
//                    min = l;
//            // Put found minimum element in its place
//            unsigned short  temp = surround[tid][i];
//            surround[tid][i] = surround[tid][min];
//            surround[tid][min] = temp;
//        }//bubble short  end
//
//        //////insertion sort start   //
//
//        /*int t,j,i;
//        for ( i = 1 ; i< MEDIAN_LENGTH ; i++) {
//            j = i;
//            while ( j > 0 && surround[tid][j] < surround[tid][j-1]) {
//                t= surround[tid][j];
//                surround[tid][j]= surround[tid][j-1];
//                surround[tid][j-1] = t;
//                j--;
//            }
//        }*/
//
//        ////insertion sort end   
//
//
//
//        * (Device_ImageData + (y * Image_Width) + x) = surround[tid][Half_Of_MEDIAN_LENGTH - 1];   // it will give value of surround[tid][4] as Median Value if use 3 x 3 matrix
//        __syncthreads();
//    }
//}
//
//int main(int argc, const char** argv)
//{
//    int dataLength;
//    int p1;
//    unsigned short* Host_ImageData = NULL;
//    ifstream is; // Read File 
//    is.open("D:\\Image_To_Be_Filtered.raw", ios::binary);
//
//    // get length of file:
//    is.seekg(0, ios::end);
//    dataLength = is.tellg();
//    is.seekg(0, ios::beg);
//
//    Host_ImageData = new  unsigned short[dataLength * sizeof(char) / sizeof(unsigned short)];
//    is.read((char*)Host_ImageData, dataLength);
//    is.close();
//
//    int Image_Width = 1580;
//    int Image_Height = 1050;
//
//    unsigned short* Host_ResultData = (unsigned short*)malloc(dataLength);
//    unsigned short* Device_ImageData = NULL;
//
//    /////////////////////////////
//    // As First time cudaMalloc take more time  for memory alocation, i dont want to cosider this time in my process. 
//    //So Please Ignore Code For Displaying First CudaMelloc Time
//    clock_t begin = clock();
//    unsigned short* forFirstCudaMalloc = NULL;
//    cudaMalloc((void**)&forFirstCudaMalloc, dataLength * sizeof(unsigned short));
//    clock_t end = clock();
//    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    cout << "First CudaMelloc time = " << elapsed_secs << "  Second\n";
//    cudaFree(forFirstCudaMalloc);
//    ////////////////////////////
//
//    //Actual Process Starts From Here 
//    clock_t beginOverAll = clock();   //
//    cudaMalloc((void**)&Device_ImageData, dataLength * sizeof(unsigned short));
//    cudaMemcpy(Device_ImageData, Host_ImageData, dataLength, cudaMemcpyHostToDevice);// copying Host Data To Device Memory For Filtering
//
//    int x = static_cast<int>(ceilf(static_cast<float>(1580.0) / BLOCK_WIDTH));
//    int y = static_cast<int>(ceilf(static_cast<float>(1050.0) / BLOCK_HEIGHT));
//
//    const dim3 grid(x, y, 1);
//    const dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
//
//    begin = clock();
//
//    MedianFilter_gpu << <grid, block >> > (Device_ImageData, Image_Width, Image_Height);
//    cudaDeviceSynchronize();
//
//    end = clock();
//    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    cout << "Process time = " << elapsed_secs << "  Second\n";
//
//    cudaMemcpy(Host_ResultData, Device_ImageData, dataLength, cudaMemcpyDeviceToHost); // copying Back Device Data To Host Memory To write In file After Filter Done
//
//    clock_t endOverall = clock();
//    elapsed_secs = double(endOverall - beginOverAll) / CLOCKS_PER_SEC;
//    cout << "Complete Time  = " << elapsed_secs << "  Second\n";
//
//    ofstream of2;   //Write Filtered Image Into File
//    of2.open("D:\\Filtered_Image.raw", ios::binary);
//    of2.write((char*)Host_ResultData, dataLength);
//    of2.close();
//    cout << "\nEnd of Writing File.  Press Any Key To Exit..!!";
//    cudaFree(Device_ImageData);
//    delete Host_ImageData;
//    delete Host_ResultData;
//
//    getch();
//    return 0;
//}