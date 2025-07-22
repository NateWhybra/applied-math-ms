// AMATH 483-583 Row Major Matrix class template starter.
// Write the methods for:
// transpose
// infinityNorm
// operator*
// operator+

#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>

template <typename T>
class Matrix
{
public:
    Matrix(int numRows, int numCols)
        : num_rows(numRows), num_cols(numCols), data(numRows * numCols) {}

    T &operator()(int i, int j)
    {
        return data[i * num_cols + j];
    }

    const T &operator()(int i, int j) const
    {
        return data[i * num_cols + j];
    }

    Matrix<T> operator*(const Matrix<T> &other) const
    {
        // Check matrix dimesions.
        if(this->num_rows != other.num_rows) {
            throw std::invalid_argument("Row dimensions do not match for element-wise multiplication.");
        }
        if(this->num_cols != other.num_cols) {
            throw std::invalid_argument("Column dimensions do not match for element-wise multiplication.");
        }

        // Do elementwise multiplication.
        Matrix<T> output(this->num_rows, this->num_cols);
        for(int i = 0; i < this->num_rows*this->num_cols; i++) {
            output.data[i] = this->data[i] * other.data[i];
        }

        return output;
    }

    Matrix<T> operator+(const Matrix<T> &other) const 
    {
        // Check matrix dimesions.
        if(this->num_rows != other.num_rows) {
            throw std::invalid_argument("Row dimensions do not match for element-wise addition.");
        }
        if(this->num_cols != other.num_cols) {
            throw std::invalid_argument("Column dimensions do not match for element-wise addition.");
        }

        // Do elementwise addition.
        Matrix<T> output(this->num_rows, this->num_cols);
        for(int i = 0; i < this->num_rows*this->num_cols; i++) {
            output.data[i] = this->data[i] + other.data[i];
        }

        return output;
    }

    Matrix<T> transpose() const
    {
        // Write your code.
        Matrix<T> A_trans(this->num_cols, this->num_rows);
        // Do tranpose.
        for(int i = 0; i < this->num_rows; i++) {
            for(int j = 0; j < this->num_cols; j++) {
                A_trans(j, i) = (*this)(i, j);
            }
        }

        return A_trans;
    }

    int numRows() const
    {
        return num_rows;
    }

    int numCols() const
    {
        return num_cols;
    }

    T infinityNorm() const
    {
        T norm = 0;
        // Write your code.
        // Infinity norm is max row sum.
        for(int i = 0; i < this->num_rows; i++) {
            // Sum the row.
            T sum = 0;
            for(int j = 0; j < this->num_cols; j++) {
                sum += std::abs((*this)(i, j));
            }

            // If it's the first row, set it to be the max.
            if(i == 0) {
                norm = sum;
            }

            // If the sum is bigger than the current max, save a new max.
            if(sum > norm) {
                norm = sum;
            } 
        }
        return norm;
    }

private:
    int num_rows;
    int num_cols;
    std::vector<T> data;
};
