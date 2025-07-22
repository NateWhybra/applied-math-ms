# include <iostream>
# include <vector>
# include <stdexcept>
# include <chrono>
# include "hw3_utils.hpp"

using namespace std;

void dgemv(double a, const vector<vector<double>> &A, vector<double> &x, double b, vector<double> &y) {
    // A is m x n and x is n x 1.
    if (A[0].size() != x.size()) {
        throw invalid_argument("Matrix A and Vector x have incompatible dimensons for multiplication.");
    }
    // A is m x n and y is m x 1.
    if (A.size() != y.size()) {
        throw invalid_argument("Matrix A and Vector y have incompatible dimensons for multiplication.");
    }

    // Initialize temporary variable.
    double current_val = 0.0;
    for(int i = 0; i < A.size(); i++) { 
        // ...
        for(int j = 0; j < A[0].size(); j++) {
            current_val += A[i][j] * x[j]; 
        }
        y[i] = a * current_val + b * y[i];
        current_val = 0.0;
    }
}