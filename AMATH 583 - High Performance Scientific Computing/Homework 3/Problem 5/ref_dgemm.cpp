# include <iostream>
# include <vector>
# include <stdexcept>
# include <chrono>
# include "hw3_utils.hpp"

using namespace std;

void dgemm(double a, const vector<vector<double>> &A, const vector<vector<double>> &B, double b, 
     vector<vector<double>> &C) {
        
     // A is m x p and B is p x n.
     // Check if A and B are compatible.
     if (A[0].size() != B.size()) {
        throw invalid_argument("Matrix A and Matrix B have incompatible dimensons.");
    }

    // Check if (A, B) and C are compatible.
    if (A.size() != C.size())  {
        throw invalid_argument("Matrix A and Matrix C have incompatible dimensons.");
    }
    if (B[0].size() != C[0].size()) {
        throw invalid_argument("Matrix B and Matrix C have incompatible dimensons.");
    }

    // For each row of C.
    for(int i = 0; i < C.size(); i++) { 
        // For each column of C.
        for(int j = 0; j < C[0].size(); j++) {
            // Initialize temporary variable.
            double current_val = 0.0;
            // Compute the matrix product AB[i, j].
            for(int k = 0; k < A[0].size(); k++) {
                current_val += A[i][k] * B[k][j];
            } 
            // C = a*(AB) + b*(C).
            C[i][j] = a * current_val + b * C[i][j];
        }
    }
}