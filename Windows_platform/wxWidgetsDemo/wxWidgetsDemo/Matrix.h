#pragma once


#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <chrono>

class Matrix {
public:
	Matrix(int, int);
	Matrix(double**, int, int);
	Matrix();
	~Matrix();
	Matrix(const Matrix&);

	inline double& operator()(int x, int y) { return p[x][y]; }

	Matrix& operator+=(const Matrix&);
	Matrix& operator-=(const Matrix&);
	Matrix& operator*=(Matrix&);
	Matrix& operator*=(double);
	Matrix& operator/=(double);
	Matrix& operator=(const Matrix&);
	//Matrix operator^(int);

	friend std::ostream& operator<<(std::ostream&, const Matrix&);
	friend std::istream& operator>>(std::istream&, Matrix&);

	void swapRows(int, int);
	Matrix transpose();
	static Matrix createIdentity(int);
	static Matrix solve(Matrix, Matrix);
	static Matrix bandSolve(Matrix, Matrix, int);

	static double dotProduct(Matrix, Matrix);

	static Matrix augment(Matrix, Matrix);
	Matrix gaussianEliminate();
	Matrix rowReduceFromGaussian();
	Matrix inverse();
	void readSolutionsFromRREF(std::ostream& os);
	void generate_random();
	std::string get_presentation();
	std::chrono::duration<double, std::milli> fp_ms;
	int get_num_rows() {
		return rows_;
	}
	int get_num_cols() {
		return cols_;
	}
	//std::string time;
private:
	int rows_, cols_;
	double** p;
	//std::string presentation;
	
	//double temp_time;

	void allocSpace();
	//Matrix expHelper(const Matrix&, int);
};

Matrix operator+(const Matrix&, const Matrix&);
Matrix operator-(const Matrix&, const Matrix&);
Matrix operator*(const Matrix&, Matrix&);
Matrix operator/(const Matrix&, const Matrix&);
Matrix operator*(double, const Matrix&);


#endif // !MATRIX_H

