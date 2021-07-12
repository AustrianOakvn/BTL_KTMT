//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
////#include <opencv2/cudaimgproc.hpp>
//#include <iostream>
//
//__host__ __device__ void swap(int* a, int* b)
//{
//	int t = *a;
//	*a = *b;
//	*b = t;
//}
//__host__ __device__ int partition(int arr[], int low, int high)
//{
//	int pivot = arr[high];    // pivot
//	int i = (low - 1);  // Index of smaller element
//
//	for (int j = low; j <= high - 1; j++)
//	{
//
//		if (arr[j] <= pivot)
//		{
//			i++;    // increment index of smaller element
//			swap(&arr[i], &arr[j]);
//		}
//	}
//	swap(&arr[i + 1], &arr[high]);
//	return (i + 1);
//}
//
///* The main function that implements QuickSort
// arr[] --> Array to be sorted,
//  low  --> Starting index,
//  high  --> Ending index */
//__host__ __device__ void quickSort(int arr[], int low, int high)
//{
//	if (low < high)
//	{
//		/* pi is partitioning index, arr[p] is now
//		   at right place */
//		int pi = partition(arr, low, high);
//
//		// Separately sort elements before
//		// partition and after partition
//		quickSort(arr, low, pi - 1);
//		quickSort(arr, pi + 1, high);
//	}
//}
//
//
//__global__ void median_filter(int* input_image, int* output_image, int image_width, int image_height, int kernel_size) {
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	//std::vector<int> kernel(kernel_size, 0);
//	int* kernel = new int[kernel_size*kernel_size];
//	
//	if (row == 0 || col == 0 || row == image_height - 1 || col == image_width - 1) {
//		output_image[row * image_width + col] = 0;
//	}
//	else {
//		for (int i = 0; i < kernel_size; i++) {
//			for (int j = 0; j < kernel_size; j++) {
//				kernel[i * kernel_size + j] = input_image[(row + i - 1) * image_width + (col + j - 1)];
//			}
//		}
//		//std::sort(kernel.begin(), kernel.end());
//		//std::sort(kernel, kernel + kernel_size - 1);
//		quickSort(kernel, 0, kernel_size);
//		output_image[row * image_width + col] = kernel[4];
//		/*std::cout << kernel[4];*/
//		
//	}
//}
//
//void cvImgToArray(const cv::Mat& mat, int *arr, int row, int col) {
//	/*size_t arr_size = row * col * sizeof(unsigned char);
//	arr = new unsigned char[arr_size];*/
//	for (int i = 0; i < row; i++) {
//		for (int j = 0; j < col; j++) {
//			arr[i * row + j] = int(mat.at<unsigned char>(i, j));
//			
//		}
//	}
//	std::cout << int(arr[0]);
//	std::cout << int(mat.at<unsigned char>(0, 0));
//	//std::cout << arr;
//}
//
//
//int main()
//{
//	/*Mat image = Mat::zeros(300, 600, CV_8UC3);
//	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
//	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
//	imshow("Display Window", image);
//	waitKey(0);
//	return 0;*/
//	//std::string image_path = "D:\\pythonProject\\Salt_Peper_Noise\\lena_noisy512.bmp";
//	std::string image_path = "C:\\Users\\AustrianOak\\Pictures\\test_2.bmp";
//
//	std::string output_path = "D:\\pythonProject\\Salt_Peper_Noise\\example.bmp";
//	cv::Mat img = imread(image_path, cv::IMREAD_GRAYSCALE);
//	if (img.empty()) {
//		std::cout << "Cout not rad the image" << image_path << std::endl;
//		return 1;
//	}
//	int img_size = img.rows * img.cols;
//	int* img_arr = new int [img_size];
//	int* img_arr_devices;
//	int* output_img;
//	int* output_img_host;
//	cvImgToArray(img, img_arr, img.rows, img.cols);
//	cudaMalloc(&img_arr_devices, img_size);
//	cudaMalloc(&output_img, img_size);
//	cudaMemcpy(img_arr_devices, img_arr, img_size, cudaMemcpyHostToDevice);
//	output_img_host = new int[img_size];
//	int threads = 16;
//	int blocks = (img_size + threads - 1) / threads;
//
//	dim3 NUM_THREADS(threads, threads);
//	dim3 NUM_BLOCKS(blocks, blocks);
//	int kernel_size = 3;
//	//std::cout << img_arr;
//	median_filter << <NUM_BLOCKS, NUM_THREADS >> > (img_arr_devices, output_img, img.rows, img.cols, kernel_size);
//	cudaDeviceSynchronize();
//	cudaMemcpy(output_img_host, output_img, img_size, cudaMemcpyDeviceToHost);
//	std::cout << int(output_img_host[0]);
//	cv::Mat result(img.rows, img.cols, CV_8UC1, output_img_host);
//	cv::imwrite(output_path, result);
//	
//	imshow("Display window", result);
//	//std::cout << img << std::endl;
//	int k = cv::waitKey(0);
//	if (k == 's') {
//		cv::imwrite("sth.jpg", img);
//	}
//	
//	return 0;
//}