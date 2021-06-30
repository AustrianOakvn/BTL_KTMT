#include <fstream>
#include <sstream>
#include <iostream>
#include "Image_processing.h"


Image::Image(std::string image_path) :image_path(image_path) {

}

void Image::read_image() {
	std::ifstream infile(image_path);
	std::stringstream ss;
	std::string inputLine = "";

	std::getline(infile, inputLine);
	if (inputLine.compare("P2") != 0) std::cerr << "Version error" << std::endl;
	else std::cout << "Version : " << inputLine << std::endl;

	ss << infile.rdbuf();
	ss >> cols >> rows >> MAX;

	int row = 0, col = 0;
	for (row = 0; row <= rows; ++row)
		array[row][0] = 0;
	for (col = 0; col <= cols; ++col)
		array[0][col] = 0;

	for (row = 1; row <= rows; ++row) {
		for (col = 1; col <= cols; ++col) {
			ss >> array[row][col];
		}
	}
}

