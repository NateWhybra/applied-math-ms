# include <iostream>
# include <vector>
# include <stdexcept>
# include <chrono>
// # include "hw3_utils.hpp"

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

// int main() {
//     // Timer foo.
//     auto start = std::chrono::high_resolution_clock::now();
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
//     long double elapsed_time = 0.L;
//     long double avg_time;
//     const int ntrials = 100;
//     vector<double> results(511);
//     vector<double> indices(511);    

//     // Loop on problem size.
//     for(int i = 2; i <= 512; i++) {
//         // Make empty vector y.
//          vector<double> y(i);
//          // Make random matrix / vector.
//          vector<vector<double>> A = random_matrix(i, i, 0, 1);
//          vector<double> x = random_vector(i, 0, 1);
//          // Define a, b.
//          double a = 2.0;
//          double b = 8.0;
//          // Augment indices.
//          indices[i-2] = i;

//         // Perform an experiment.
//         for (int t = 0; t < ntrials; t++) {
//             start = std::chrono::high_resolution_clock::now();
            
//             // Do work (size i, trial t).
//             dgemv(a, A, x, b, y);
            
//             stop = std::chrono::high_resolution_clock::now();
//             duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
//             elapsed_time += (duration.count() * 1.e-9); // Convert duration to seconds.
//         }
//         avg_time = elapsed_time / static_cast<long double>(ntrials);
           
//         // Save or report findings.
//         results[i-2] = avg_time;
//         string data_name = "dgemv_results";
//         // Save data to .csv.
//         vec_to_csv(results, data_name);
        
//         // Zero time again.
//         elapsed_time = 0.L;
//     }
// }