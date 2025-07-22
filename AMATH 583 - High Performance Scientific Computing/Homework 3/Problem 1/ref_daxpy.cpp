# include <iostream>
# include <vector>
# include <stdexcept>
# include <chrono>
// # include "hw3_utils.hpp"
using namespace std;

void daxpy(double a, const vector<double> &x, vector<double> &y) {
    // Check that the vectors are the same size.
    if (x.size() != y.size()) {
        throw invalid_argument("Vectors must be the same length.");
    }
    
    // Otherwise, do the daxpy.
    for(int i = 0; i < x.size(); i++) {
        y[i] += a * x[i];
    }
}

// int main() {
//     // Timer foo.
//     auto start = std::chrono::high_resolution_clock::now();
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
//     long double elapsed_time = 0.L ;
//     long double avg_time ;
//     const int ntrials = 100;
//     vector<double> results(511);


//     // Loop on problem size.
//     for(int i = 2; i <= 512; i++) {
//         // Make empty vector y.
//          vector<double> y(i);
//          // Make random matrix.
//          vector<double> x = random_vector(i, 0, 1);
//          // Define a.
//          double a = 2.0;
        
//         // Perform an experiment.
//         for (int t = 0; t < ntrials; t++) {
//             start = std::chrono::high_resolution_clock::now();
            
//             // Do work (size i, trial t).
//             daxpy(a, x, y);
            
//             stop = std::chrono::high_resolution_clock::now();
//             duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
//             elapsed_time += (duration.count() * 1.e-9); // Convert duration to seconds.
//         }
//         avg_time = elapsed_time / static_cast<long double>(ntrials);

//         // Print.
//         cout << "Problem size = " << i << ", Avg time = " << avg_time << " seconds" << endl;
           
//         // Save or report findings.
//         results[i-2] = avg_time;
        
//         // Zero time again.
//         elapsed_time = 0.L;
//     }

//     string data_name = "daxpy_results";
//     // Save data to .csv.
//     vec_to_csv(results, data_name);
// }