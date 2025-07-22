# include <iostream>
# include <vector>
# include <stdexcept>
# include <chrono>
// # include "hw3_utils.hpp"

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

// int main() {
//     // Timer foo.
//     auto start = std::chrono::high_resolution_clock::now();
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
//     long double elapsed_time = 0.L;
//     long double avg_time;
//     const int ntrials = 10;
    
//     vector<double> results(511);
//     vector<double> indices(511);    

//     // Loop on problem size.
//     for(int i = 2; i <= 512; i++) {
//         // Make empty matrix C.
//          vector<vector<double>> C(i, vector<double>(i));
         
//          // Make random matrices A and B.
//          vector<vector<double>> A = random_matrix(i, i, 0, 1);
//          vector<vector<double>> B = random_matrix(i, i, 0, 1);
         
//          // Define a, b.
//          double a = 2.0;
//          double b = 8.0;
         
//          // Augment indices.
//          indices[i-2] = i;

//         // Perform an experiment.
//         for (int t = 0; t < ntrials; t++) {
//             start = std::chrono::high_resolution_clock::now();
            
//             // Do work (size i, trial t).
//             dgemm(a, A, B, b, C);
            
//             stop = std::chrono::high_resolution_clock::now();
//             duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
//             elapsed_time += (duration.count() * 1.e-9); // Convert duration to seconds.
//         }
//         avg_time = elapsed_time / static_cast<long double>(ntrials);
           
//         // Save or report findings.
//         results[i-2] = avg_time;
//         string data_name = "dgemm_results";
        
//         // Save data to .csv.
//         vec_to_csv(results, data_name);
        
//         // Zero time again.
//         elapsed_time = 0.L;
//     }
// }