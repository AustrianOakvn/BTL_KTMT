#pragma once
#include <string>


class Image {
private:
	std::string image_path;
	int rows, cols, MAX;
	int window[9];
	int array[2000][2000], arr[2000][2000];
public:
	Image(std::string);
	void read_image();
	void perform_median_filter();
};